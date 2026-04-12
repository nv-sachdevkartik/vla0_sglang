#!/usr/bin/env python3
"""
Fast VLA-0 Inference V2 - Focused on Action Regression Head
Eliminates the autoregressive bottleneck (208 tokens @ ~20ms/tok = 4.3s)
Target: 4 Hz (250ms)

Key insight: The backbone forward pass is ~100ms. The generate loop is ~4200ms.
By replacing text generation with a regression head, we get 56 action values
from a single forward pass.
"""
import os
import sys
import time
import json
import gc
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Setup paths
sys.path.insert(0, '/home/shadeform/vla0')
os.chdir('/home/shadeform/vla0')

# Mock lerobot metadata
try:
    import roboverse.datasets.lerobot.dataloader as _rvlr
    class _MockMetadata:
        camera_keys = ['image', 'wrist_image']
    _rvlr.get_lerobot_metadata = lambda repo_id: _MockMetadata()
except:
    pass

CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'
RESULTS_DIR = Path('/home/shadeform/vla0-compression/results')

def log(msg):
    print(f"[{datetime.utcnow().strftime('%H:%M:%S')}] {msg}", flush=True)


def get_dummy_inputs(batch_size=1, device='cuda'):
    """Create dummy inputs matching VLA-0 format."""
    # Shape: (batch, history=1, num_cam=2, H=224, W=224, C=3)
    rgb = torch.randint(0, 255, (batch_size, 1, 2, 224, 224, 3), dtype=torch.uint8).float().to(device)
    instr = ["pick up the red block and place it in the basket"] * batch_size
    return rgb, instr


def benchmark(fn, warmup=5, iterations=20, name=""):
    """Benchmark a function, return Hz and ms."""
    # Warmup
    for _ in range(warmup):
        fn()
    
    # Benchmark
    torch.cuda.synchronize()
    times = []
    for _ in range(iterations):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    
    mean_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    hz = 1000 / mean_ms
    log(f"{name}: {hz:.2f} Hz ({mean_ms:.1f} ± {std_ms:.1f} ms)")
    return {"hz": hz, "mean_ms": mean_ms, "std_ms": std_ms}


def load_model():
    """Load the VLA-0 model."""
    from rv_train.train import get_pretrained_model
    log("Loading model...")
    model, cfg = get_pretrained_model(CKPT, device='cuda', torch_compile=False)
    model.eval()
    log(f"Model loaded. Params: {sum(p.numel() for p in model.parameters())/1e9:.2f}B")
    return model, cfg


class ActionRegressionHead(nn.Module):
    """
    MLP head that predicts 56 action values directly from the last hidden state.
    Replaces autoregressive text generation with a single forward pass.
    """
    def __init__(self, hidden_size=2048, num_outputs=56, hidden_dim=512):
        super().__init__()
        self.num_outputs = num_outputs
        self.norm = nn.LayerNorm(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_outputs),
        )
        
    def forward(self, hidden_states, attention_mask=None):
        """
        hidden_states: (batch, seq_len, hidden_size)
        Returns: (batch, num_outputs) normalized to [0, 1]
        """
        # Use the last non-padding token's hidden state
        if attention_mask is not None:
            # Find last valid position
            seq_lengths = attention_mask.sum(dim=1) - 1
            batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
            last_hidden = hidden_states[batch_indices, seq_lengths.long()]
        else:
            last_hidden = hidden_states[:, -1]
        
        x = self.norm(last_hidden)
        x = self.head(x)
        return torch.sigmoid(x)  # Output in [0, 1], scale to action range later


def run_backbone_only_benchmark(model, rgb, instr):
    """
    Measure backbone forward pass without text generation.
    This is the theoretical lower bound for any approach.
    """
    log("\n=== BACKBONE-ONLY BENCHMARK ===")
    
    # Prepare inputs once
    bs = len(instr)
    imgs = model.get_imgs(bs=bs, pc=None, rgb_pc=None, rgb=rgb)
    model_inputs, _ = model.get_qwen_inputs(
        bs=bs, imgs=imgs, instr=instr,
        action_txt=[[]] * bs,
        drop_assistant=True,
        add_generation_prompt=True,
    )
    
    log(f"Input sequence length: {model_inputs['input_ids'].shape[1]} tokens")
    
    # Measure prefill-only (backbone forward)
    def backbone_fn():
        with torch.no_grad():
            outputs = model.model.model(
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs['attention_mask'],
                pixel_values=model_inputs.get('pixel_values'),
                image_grid_thw=model_inputs.get('image_grid_thw'),
                output_hidden_states=True,
                return_dict=True,
            )
        return outputs.last_hidden_state
    
    result = benchmark(backbone_fn, warmup=3, iterations=15, name="Backbone only (prefill)")
    
    return result, model_inputs


def run_regression_head_benchmark(model, rgb, instr, model_inputs):
    """
    Benchmark the full regression pipeline:
    1. Backbone forward pass
    2. Action head MLP
    """
    log("\n=== REGRESSION HEAD BENCHMARK ===")
    
    # Get hidden size
    config = model.model.config
    text_config = config.text_config if hasattr(config, 'text_config') else config
    hidden_size = text_config.hidden_size
    
    # Create regression head
    action_head = ActionRegressionHead(
        hidden_size=hidden_size,
        num_outputs=56,  # 8 timesteps * 7 dims
        hidden_dim=512,
    ).cuda().to(torch.bfloat16)
    action_head.eval()
    
    # Get dataset stats for denormalization
    min_act = torch.tensor(model.dataset_stats['min'], device='cuda')
    max_act = torch.tensor(model.dataset_stats['max'], device='cuda')
    
    def regression_fn():
        with torch.no_grad():
            # Backbone forward
            outputs = model.model.model(
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs['attention_mask'],
                pixel_values=model_inputs.get('pixel_values'),
                image_grid_thw=model_inputs.get('image_grid_thw'),
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            
            # Action head
            action_normalized = action_head(hidden_states, model_inputs['attention_mask'])
            
            # Denormalize
            actions = action_normalized * (max_act - min_act) + min_act
            return actions.view(1, 8, 7)
    
    result = benchmark(regression_fn, warmup=3, iterations=15, name="Backbone + Regression head")
    
    return result, action_head


def run_baseline_benchmark(model, rgb, instr):
    """Baseline full autoregressive generation."""
    log("\n=== BASELINE (AUTOREGRESSIVE) BENCHMARK ===")
    
    def baseline_fn():
        with torch.no_grad():
            return model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
    
    result = benchmark(baseline_fn, warmup=2, iterations=10, name="Baseline autoregressive")
    return result


def run_reduced_horizon_benchmark(model, rgb, instr):
    """
    Benchmark with reduced horizon (2 instead of 8).
    This is a simple change that requires no retraining.
    """
    log("\n=== REDUCED HORIZON BENCHMARK ===")
    
    # Save original values
    orig_horizon = model.horizon
    orig_system_message = model.system_message
    
    # Set reduced horizon
    horizon = 2
    model.horizon = horizon
    model.system_message = f"Analyze the input image and predict robot actions for the next {horizon} timesteps. Each action has {model.act_dim} dimensions. Output a single sequence of {horizon * model.act_dim} integers (0-{model.num_bins_actions} each), representing the {horizon} timesteps sequentially. Provide only space separated numbers. Nothing else."
    
    def reduced_fn():
        with torch.no_grad():
            return model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
    
    try:
        result = benchmark(reduced_fn, warmup=2, iterations=10, name="Horizon=2 (from 8)")
        
        # Check output
        with torch.no_grad():
            out = model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
            log(f"  Generated text: {out['pred_action_txt'][0]}")
            log(f"  Action shape: {out['out_ori_act'].shape}")
    except Exception as e:
        log(f"Reduced horizon failed: {e}")
        result = {"hz": 0, "error": str(e)}
    finally:
        # Restore original values
        model.horizon = orig_horizon
        model.system_message = orig_system_message
    
    return result


def distill_action_head(model, action_head, rgb_samples, instr_samples, epochs=100, lr=1e-3):
    """
    Train the action head to match teacher model outputs.
    """
    log(f"\n=== DISTILLATION ({epochs} epochs, {len(rgb_samples)} samples) ===")
    
    # Get dataset stats for normalization
    min_act = torch.tensor(model.dataset_stats['min'], device='cuda')
    max_act = torch.tensor(model.dataset_stats['max'], device='cuda')
    
    # Collect teacher outputs and hidden states
    teacher_actions = []
    hidden_states_list = []
    attention_masks = []
    
    model.eval()
    log("Collecting teacher outputs...")
    for i, (rgb, instr) in enumerate(zip(rgb_samples, instr_samples)):
        with torch.no_grad():
            # Get teacher action via text generation
            out = model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
            teacher_actions.append(out['out_ori_act'])
            
            # Get hidden states
            imgs = model.get_imgs(bs=1, pc=None, rgb_pc=None, rgb=rgb)
            model_inputs, _ = model.get_qwen_inputs(
                bs=1, imgs=imgs, instr=instr,
                action_txt=[[]], drop_assistant=True, add_generation_prompt=True
            )
            
            outputs = model.model.model(
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs['attention_mask'],
                pixel_values=model_inputs.get('pixel_values'),
                image_grid_thw=model_inputs.get('image_grid_thw'),
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states_list.append(outputs.last_hidden_state.detach())
            attention_masks.append(model_inputs['attention_mask'].detach())
        
        if (i + 1) % 5 == 0:
            log(f"  Collected {i+1}/{len(rgb_samples)}")
    
    # Stack teacher actions and normalize to [0, 1]
    all_teacher_actions = torch.cat(teacher_actions, dim=0)  # (N, 8, 7)
    all_teacher_flat = all_teacher_actions.view(-1, 56)  # (N, 56)
    all_teacher_normalized = (all_teacher_flat - min_act.view(-1)) / (max_act - min_act).view(-1)
    
    # Train action head
    optimizer = torch.optim.AdamW(action_head.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    
    action_head.train()
    log("Training action head...")
    for epoch in range(epochs):
        total_loss = 0
        for i, (hidden, mask) in enumerate(zip(hidden_states_list, attention_masks)):
            optimizer.zero_grad()
            pred = action_head(hidden, mask)
            target = all_teacher_normalized[i:i+1]
            loss = F.mse_loss(pred, target)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        scheduler.step()
        if (epoch + 1) % 20 == 0:
            log(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(hidden_states_list):.6f}")
    
    action_head.eval()
    
    # Evaluate quality
    log("Evaluating distillation quality...")
    total_mse = 0
    total_mae = 0
    with torch.no_grad():
        for i, (hidden, mask) in enumerate(zip(hidden_states_list, attention_masks)):
            pred = action_head(hidden, mask)
            target = all_teacher_normalized[i:i+1]
            total_mse += F.mse_loss(pred, target).item()
            total_mae += F.l1_loss(pred, target).item()
    
    log(f"  Mean MSE: {total_mse/len(hidden_states_list):.6f}")
    log(f"  Mean MAE: {total_mae/len(hidden_states_list):.4f}")
    
    return action_head


def main():
    log("=" * 60)
    log("VLA-0 Fast Inference V2 - Regression Head Focus")
    log("Target: 4 Hz (250ms)")
    log("=" * 60)
    
    # Load model
    model, cfg = load_model()
    rgb, instr = get_dummy_inputs()
    
    results = {}
    
    # 1. Baseline benchmark
    results['baseline'] = run_baseline_benchmark(model, rgb, instr)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # 2. Reduced horizon benchmark (simple, no retraining)
    results['horizon_2'] = run_reduced_horizon_benchmark(model, rgb, instr)
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # 3. Backbone-only benchmark (theoretical lower bound)
    backbone_result, model_inputs = run_backbone_only_benchmark(model, rgb, instr)
    results['backbone_only'] = backbone_result
    
    # 4. Regression head benchmark (random weights)
    regression_result, action_head = run_regression_head_benchmark(model, rgb, instr, model_inputs)
    results['regression_random'] = regression_result
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # 5. Distillation and benchmark
    log("\n=== DISTILLATION EXPERIMENT ===")
    
    # Generate diverse calibration samples (vary instruction slightly)
    instructions = [
        "pick up the red block and place it in the basket",
        "move the blue cube to the left",
        "grasp the green object and lift it",
        "push the yellow item forward",
        "place the object on the table",
        "pick up the apple and put it in the bowl",
        "grab the bottle and move it right",
        "take the spoon and place it nearby",
        "lift the cup and set it down gently",
        "move the plate to the center",
    ]
    
    rgb_samples = [rgb.clone() for _ in range(10)]
    instr_samples = [[inst] for inst in instructions]
    
    # Distill
    action_head = distill_action_head(model, action_head, rgb_samples, instr_samples, epochs=100, lr=1e-3)
    
    # Re-benchmark with trained head
    log("\n=== POST-DISTILLATION BENCHMARK ===")
    
    # Get dataset stats for denormalization
    min_act = torch.tensor(model.dataset_stats['min'], device='cuda')
    max_act = torch.tensor(model.dataset_stats['max'], device='cuda')
    
    def distilled_regression_fn():
        with torch.no_grad():
            # Backbone forward
            outputs = model.model.model(
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs['attention_mask'],
                pixel_values=model_inputs.get('pixel_values'),
                image_grid_thw=model_inputs.get('image_grid_thw'),
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = outputs.last_hidden_state
            
            # Action head
            action_normalized = action_head(hidden_states, model_inputs['attention_mask'])
            
            # Denormalize
            actions = action_normalized * (max_act - min_act) + min_act
            return actions.view(1, 8, 7)
    
    results['regression_distilled'] = benchmark(distilled_regression_fn, warmup=3, iterations=15, name="Distilled regression head")
    
    # Compare quality
    log("\n=== QUALITY COMPARISON ===")
    with torch.no_grad():
        teacher_out = model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
        student_out = distilled_regression_fn()
        
        teacher_act = teacher_out['out_ori_act']
        
        mse = F.mse_loss(student_out, teacher_act).item()
        mae = F.l1_loss(student_out, teacher_act).item()
        log(f"Teacher vs Student: MSE={mse:.6f}, MAE={mae:.4f}")
        log(f"Teacher first timestep: {teacher_act[0, 0].cpu().numpy()}")
        log(f"Student first timestep: {student_out[0, 0].cpu().numpy()}")
    
    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    
    for name, data in results.items():
        if isinstance(data, dict) and 'hz' in data:
            hz = data['hz']
            ms = data.get('mean_ms', 0)
            log(f"{name}: {hz:.2f} Hz ({ms:.1f} ms)")
    
    best_hz = max(r['hz'] for r in results.values() if isinstance(r, dict) and 'hz' in r)
    log(f"\nBest achieved: {best_hz:.2f} Hz")
    log(f"Target: 4.0 Hz (250ms)")
    log(f"Gap to target: {4.0 - best_hz:.2f} Hz")
    
    # Speedup calculations
    baseline_ms = results['baseline']['mean_ms']
    regression_ms = results['regression_distilled']['mean_ms']
    log(f"\nSpeedup from baseline: {baseline_ms / regression_ms:.1f}x")
    
    # Save results
    results_path = RESULTS_DIR / 'fast_inference_v2_results.json'
    with open(results_path, 'w') as f:
        # Convert any non-serializable items
        def sanitize(obj):
            if isinstance(obj, dict):
                return {k: sanitize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [sanitize(v) for v in obj]
            elif isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        json.dump(sanitize(results), f, indent=2)
    log(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
