#!/usr/bin/env python3
"""
Fast VLA-0 Inference Pipeline
Eliminates the autoregressive bottleneck (208 tokens @ ~20ms/tok = 4.3s)
Target: 4 Hz (250ms)

Approaches implemented:
- C: torch.compile + Static KV Cache + CUDA Graphs
- D: Reduced token budget (fewer bins, shorter horizon)
- A: Action regression head (single forward pass → 56 floats)
"""
import os
import sys
import time
import json
import gc
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

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


# =============================================================================
# APPROACH C: torch.compile + Static Cache + CUDA Graphs
# =============================================================================

def approach_c_compile_static_cache(model, rgb, instr):
    """
    Approach C: Use torch.compile with static KV cache.
    
    Key optimizations:
    1. torch.compile(mode="reduce-overhead") on generate
    2. Static KV cache allocation (avoids dynamic allocation overhead)
    3. Pad sequences to fixed length for CUDA graph compatibility
    """
    log("\n=== APPROACH C: torch.compile + Static Cache ===")
    
    # Get baseline first
    def baseline_fn():
        with torch.no_grad():
            return model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
    
    baseline_result = benchmark(baseline_fn, warmup=3, iterations=10, name="C.0 Baseline")
    
    # Try torch.compile on the inner model's generate
    log("Compiling model.model.generate with reduce-overhead...")
    
    # First, let's try compiling just the forward pass
    original_generate = model.model.generate
    
    try:
        # Compile with reduce-overhead mode (uses CUDA graphs internally)
        compiled_generate = torch.compile(
            original_generate,
            mode="reduce-overhead",
            fullgraph=False,  # Allow graph breaks for generate's complex control flow
        )
        model.model.generate = compiled_generate
        
        def compiled_fn():
            with torch.no_grad():
                return model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
        
        compiled_result = benchmark(compiled_fn, warmup=10, iterations=20, name="C.1 torch.compile generate")
    except Exception as e:
        log(f"torch.compile failed: {e}")
        compiled_result = {"hz": 0, "mean_ms": float('inf'), "error": str(e)}
    finally:
        model.model.generate = original_generate
    
    # Try static cache
    log("Testing static KV cache...")
    try:
        from transformers import StaticCache
        
        # Get model config
        config = model.model.config
        text_config = config.text_config if hasattr(config, 'text_config') else config
        
        # Create static cache
        # Input is ~1200 tokens (image + prompt), output is ~208 tokens
        max_cache_len = 2048  # Generous upper bound
        
        def static_cache_fn():
            with torch.no_grad():
                # Prepare inputs
                imgs = model.get_imgs(bs=1, pc=None, rgb_pc=None, rgb=rgb)
                model_inputs, _ = model.get_qwen_inputs(
                    bs=1, imgs=imgs, instr=instr,
                    action_txt=[[]], drop_assistant=True, add_generation_prompt=True
                )
                
                # Create fresh static cache for this generation
                cache = StaticCache(
                    config=text_config,
                    max_cache_len=max_cache_len,
                )
                
                generated_ids = model.model.generate(
                    **model_inputs,
                    past_key_values=cache,
                    max_new_tokens=280,
                    do_sample=False,
                    logits_processor=[model.logits_processor],
                )
                
                # Parse output
                n_input = model_inputs['input_ids'].shape[1]
                text = model.processor.batch_decode(
                    generated_ids[:, n_input:], 
                    skip_special_tokens=True
                )[0]
                return model.get_action_from_text_action([text])
        
        static_cache_result = benchmark(static_cache_fn, warmup=5, iterations=15, name="C.2 Static KV Cache")
    except Exception as e:
        log(f"Static cache failed: {e}")
        import traceback
        traceback.print_exc()
        static_cache_result = {"hz": 0, "mean_ms": float('inf'), "error": str(e)}
    
    return {
        "baseline": baseline_result,
        "compiled": compiled_result,
        "static_cache": static_cache_result,
    }


# =============================================================================
# APPROACH D: Reduced Token Budget
# =============================================================================

class ReducedTokenModel(nn.Module):
    """
    Wrapper that reduces token budget by:
    1. Using fewer bins (64 instead of 1000) - each number is 1-2 tokens instead of 3-4
    2. Shorter horizon (2 instead of 8) - predict fewer timesteps
    """
    def __init__(self, base_model, num_bins=64, horizon=2):
        super().__init__()
        self.base = base_model
        self.num_bins = num_bins
        self.horizon = horizon
        self.act_dim = base_model.act_dim
        
        # Store original values
        self.orig_num_bins = base_model.num_bins_actions
        self.orig_horizon = base_model.horizon
        
        # Update system message for reduced output
        self.system_message = f"Analyze the input image and predict robot actions for the next {horizon} timesteps. Each action has {self.act_dim} dimensions. Output a single sequence of {horizon * self.act_dim} integers (0-{num_bins} each), representing the {horizon} timesteps sequentially. Provide only space separated numbers. Nothing else."
        
    def forward(self, rgb, instr, get_action=True, get_loss=False):
        # Temporarily modify model settings
        orig_system = self.base.system_message
        orig_bins = self.base.num_bins_actions
        orig_horizon = self.base.horizon
        
        self.base.system_message = self.system_message
        self.base.num_bins_actions = self.num_bins
        self.base.horizon = self.horizon
        
        try:
            result = self.base(
                rgb=rgb, instr=instr, 
                get_action=get_action, get_loss=get_loss,
                generate_temperature=0,  # Greedy
            )
            
            # Scale actions back from reduced bins to original scale
            if 'out_ori_act' in result:
                # The action parsing already uses num_bins_actions, 
                # so we need to rescale from [0, num_bins] to original range
                pass  # get_action_from_text_action handles this
            
            return result
        finally:
            self.base.system_message = orig_system
            self.base.num_bins_actions = orig_bins
            self.base.horizon = orig_horizon


def approach_d_reduced_tokens(model, rgb, instr):
    """
    Approach D: Reduce token budget.
    
    Original: 8 timesteps × 7 dims × ~4 tokens/number = ~224 tokens
    
    Options:
    D.1: 64 bins (1-2 tokens/number): 8 × 7 × 2 = 112 tokens (~50% reduction)
    D.2: Horizon 2 (repredict often): 2 × 7 × 4 = 56 tokens (~75% reduction)  
    D.3: Both: 2 × 7 × 2 = 28 tokens (~87% reduction)
    """
    log("\n=== APPROACH D: Reduced Token Budget ===")
    
    results = {}
    
    # D.1: Reduced bins (64 instead of 1000)
    log("D.1: Testing 64 bins (from 1000)...")
    try:
        reduced_bins_model = ReducedTokenModel(model, num_bins=64, horizon=8)
        
        def reduced_bins_fn():
            with torch.no_grad():
                return reduced_bins_model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
        
        results["bins_64"] = benchmark(reduced_bins_fn, warmup=3, iterations=10, name="D.1 64 bins")
        
        # Check output tokens
        with torch.no_grad():
            out = reduced_bins_model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
            log(f"    Output text: {out['pred_action_txt'][0][:100]}...")
    except Exception as e:
        log(f"D.1 failed: {e}")
        import traceback
        traceback.print_exc()
        results["bins_64"] = {"hz": 0, "error": str(e)}
    
    # D.2: Reduced horizon (2 instead of 8)
    log("D.2: Testing horizon=2 (from 8)...")
    try:
        reduced_horizon_model = ReducedTokenModel(model, num_bins=1000, horizon=2)
        
        def reduced_horizon_fn():
            with torch.no_grad():
                return reduced_horizon_model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
        
        results["horizon_2"] = benchmark(reduced_horizon_fn, warmup=3, iterations=10, name="D.2 Horizon 2")
    except Exception as e:
        log(f"D.2 failed: {e}")
        results["horizon_2"] = {"hz": 0, "error": str(e)}
    
    # D.3: Both reduced bins AND horizon
    log("D.3: Testing 64 bins + horizon=2...")
    try:
        reduced_both_model = ReducedTokenModel(model, num_bins=64, horizon=2)
        
        def reduced_both_fn():
            with torch.no_grad():
                return reduced_both_model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
        
        results["bins_64_horizon_2"] = benchmark(reduced_both_fn, warmup=3, iterations=10, name="D.3 64 bins + H=2")
        
        # Check output tokens
        with torch.no_grad():
            out = reduced_both_model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
            log(f"    Output text: {out['pred_action_txt'][0]}")
            log(f"    Action shape: {out['out_ori_act'].shape}")
    except Exception as e:
        log(f"D.3 failed: {e}")
        results["bins_64_horizon_2"] = {"hz": 0, "error": str(e)}
    
    # D.4: Extreme reduction - 32 bins, horizon 1
    log("D.4: Testing 32 bins + horizon=1 (minimum tokens)...")
    try:
        extreme_model = ReducedTokenModel(model, num_bins=32, horizon=1)
        
        def extreme_fn():
            with torch.no_grad():
                return extreme_model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
        
        results["bins_32_horizon_1"] = benchmark(extreme_fn, warmup=3, iterations=10, name="D.4 32 bins + H=1")
        
        with torch.no_grad():
            out = extreme_model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
            log(f"    Output: {out['pred_action_txt'][0]}")
    except Exception as e:
        log(f"D.4 failed: {e}")
        results["bins_32_horizon_1"] = {"hz": 0, "error": str(e)}
    
    return results


# =============================================================================
# APPROACH A: Action Regression Head (eliminates autoregressive decode entirely)
# =============================================================================

class ActionRegressionHead(nn.Module):
    """
    MLP head that predicts 56 action values directly from the last hidden state.
    Replaces autoregressive text generation with a single forward pass.
    
    Architecture: hidden_state → LayerNorm → Linear → GELU → Linear → 56 floats
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
            last_hidden = hidden_states[batch_indices, seq_lengths]
        else:
            last_hidden = hidden_states[:, -1]
        
        x = self.norm(last_hidden)
        x = self.head(x)
        return torch.sigmoid(x)  # Output in [0, 1], scale to action range later


class FastActionModel(nn.Module):
    """
    VLA-0 model with regression head instead of text generation.
    Single forward pass through backbone → MLP head → 56 action values.
    """
    def __init__(self, base_model, horizon=8, act_dim=7, num_bins=1000):
        super().__init__()
        self.base = base_model
        self.horizon = horizon
        self.act_dim = act_dim
        self.num_bins = num_bins
        self.num_outputs = horizon * act_dim  # 56
        
        # Get hidden size from model config
        config = base_model.model.config
        text_config = config.text_config if hasattr(config, 'text_config') else config
        hidden_size = text_config.hidden_size  # 2048 for Qwen2.5-VL-3B
        
        # Create regression head
        self.action_head = ActionRegressionHead(
            hidden_size=hidden_size,
            num_outputs=self.num_outputs,
            hidden_dim=512,
        ).to(next(base_model.parameters()).device).to(torch.bfloat16)
        
        # Dataset stats for denormalization
        self.min_act = torch.tensor(base_model.dataset_stats['min'], device='cuda')
        self.max_act = torch.tensor(base_model.dataset_stats['max'], device='cuda')
        
    def forward(self, rgb, instr, return_hidden=False):
        """
        Single forward pass inference.
        Returns action tensor of shape (batch, horizon, act_dim).
        """
        bs = len(instr)
        
        # Prepare inputs (same as base model)
        imgs = self.base.get_imgs(bs=bs, pc=None, rgb_pc=None, rgb=rgb)
        model_inputs, _ = self.base.get_qwen_inputs(
            bs=bs, imgs=imgs, instr=instr,
            action_txt=[[]] * bs,
            drop_assistant=True,
            add_generation_prompt=True,
        )
        
        # Forward pass through backbone (WITHOUT generation)
        with torch.no_grad():
            outputs = self.base.model.model(  # .model.model is the Qwen2_5_VLModel
                input_ids=model_inputs['input_ids'],
                attention_mask=model_inputs['attention_mask'],
                pixel_values=model_inputs.get('pixel_values'),
                image_grid_thw=model_inputs.get('image_grid_thw'),
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Get last hidden state
        hidden_states = outputs.last_hidden_state  # (batch, seq_len, hidden_size)
        
        if return_hidden:
            return hidden_states
        
        # Pass through action head
        action_normalized = self.action_head(hidden_states, model_inputs['attention_mask'])
        
        # Denormalize to action range
        # action_normalized is in [0, 1], scale to [min_act, max_act]
        action_flat = action_normalized * (self.max_act - self.min_act) + self.min_act
        
        # Reshape to (batch, horizon, act_dim)
        actions = action_flat.view(bs, self.horizon, self.act_dim)
        
        return {'out_ori_act': actions}
    
    def distill_from_teacher(self, rgb_samples, instr_samples, epochs=100, lr=1e-3):
        """
        Train the action head to match teacher model outputs.
        Uses MSE loss between regression output and teacher's decoded actions.
        """
        log(f"Distilling action head from teacher ({epochs} epochs, {len(rgb_samples)} samples)...")
        
        # Collect teacher outputs
        teacher_actions = []
        hidden_states_list = []
        attention_masks = []
        
        self.base.eval()
        with torch.no_grad():
            for rgb, instr in zip(rgb_samples, instr_samples):
                # Get teacher action via text generation
                out = self.base(rgb=rgb, instr=instr, get_action=True, get_loss=False)
                teacher_actions.append(out['out_ori_act'])
                
                # Get hidden states for this input
                hidden = self.forward(rgb, instr, return_hidden=True)
                hidden_states_list.append(hidden)
                
                # Get attention mask
                imgs = self.base.get_imgs(bs=1, pc=None, rgb_pc=None, rgb=rgb)
                model_inputs, _ = self.base.get_qwen_inputs(
                    bs=1, imgs=imgs, instr=instr,
                    action_txt=[[]], drop_assistant=True, add_generation_prompt=True
                )
                attention_masks.append(model_inputs['attention_mask'])
        
        # Stack all data
        all_teacher_actions = torch.cat(teacher_actions, dim=0)  # (N, horizon, act_dim)
        all_teacher_flat = all_teacher_actions.view(-1, self.num_outputs)  # (N, 56)
        
        # Normalize teacher actions to [0, 1]
        all_teacher_normalized = (all_teacher_flat - self.min_act) / (self.max_act - self.min_act)
        
        # Train action head
        optimizer = torch.optim.AdamW(self.action_head.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
        
        self.action_head.train()
        for epoch in range(epochs):
            total_loss = 0
            for i, (hidden, mask) in enumerate(zip(hidden_states_list, attention_masks)):
                optimizer.zero_grad()
                pred = self.action_head(hidden.detach(), mask)
                target = all_teacher_normalized[i:i+1]
                loss = F.mse_loss(pred, target)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            scheduler.step()
            if (epoch + 1) % 20 == 0:
                log(f"  Epoch {epoch+1}/{epochs}: loss={total_loss/len(hidden_states_list):.6f}")
        
        self.action_head.eval()
        log("Distillation complete.")


def approach_a_regression_head(model, rgb, instr):
    """
    Approach A: Replace text generation with direct action regression.
    
    This is the most promising approach for reaching 4 Hz:
    - Single forward pass through backbone (~50-100ms)
    - Small MLP head (~1ms)
    - No autoregressive decode at all
    
    Requires training/distillation of the MLP head.
    """
    log("\n=== APPROACH A: Action Regression Head ===")
    
    results = {}
    
    # A.1: First, measure backbone-only forward pass time
    log("A.1: Measuring backbone forward pass time...")
    
    fast_model = FastActionModel(model, horizon=8, act_dim=7)
    
    def backbone_only_fn():
        with torch.no_grad():
            return fast_model.forward(rgb, instr, return_hidden=True)
    
    results["backbone_only"] = benchmark(backbone_only_fn, warmup=5, iterations=20, name="A.1 Backbone only")
    
    # A.2: Full regression pipeline (untrained head - random output)
    log("A.2: Full regression pipeline (random head)...")
    
    def regression_fn():
        with torch.no_grad():
            return fast_model.forward(rgb, instr)
    
    results["regression_random"] = benchmark(regression_fn, warmup=5, iterations=20, name="A.2 Regression (random)")
    
    # A.3: Quick distillation test (10 samples, just to show it works)
    log("A.3: Quick distillation (10 samples)...")
    try:
        # Generate some calibration samples
        rgb_samples = [rgb.clone() for _ in range(10)]
        instr_samples = [instr for _ in range(10)]
        
        fast_model.distill_from_teacher(rgb_samples, instr_samples, epochs=50, lr=1e-3)
        
        results["regression_distilled"] = benchmark(regression_fn, warmup=5, iterations=20, name="A.3 Regression (distilled)")
        
        # Check output quality
        with torch.no_grad():
            teacher_out = model(rgb=rgb, instr=instr, get_action=True, get_loss=False)
            student_out = fast_model.forward(rgb, instr)
            
            teacher_act = teacher_out['out_ori_act']
            student_act = student_out['out_ori_act']
            
            mse = F.mse_loss(student_act, teacher_act).item()
            mae = F.l1_loss(student_act, teacher_act).item()
            log(f"    Teacher vs Student: MSE={mse:.6f}, MAE={mae:.4f}")
            log(f"    Teacher first action: {teacher_act[0, 0].cpu().numpy()}")
            log(f"    Student first action: {student_act[0, 0].cpu().numpy()}")
    except Exception as e:
        log(f"A.3 distillation failed: {e}")
        import traceback
        traceback.print_exc()
        results["regression_distilled"] = {"hz": 0, "error": str(e)}
    
    # A.4: Compile the regression model
    log("A.4: torch.compile on regression model...")
    try:
        compiled_fast_model = torch.compile(fast_model, mode="reduce-overhead")
        
        def compiled_regression_fn():
            with torch.no_grad():
                return compiled_fast_model.forward(rgb, instr)
        
        results["regression_compiled"] = benchmark(compiled_regression_fn, warmup=10, iterations=20, name="A.4 Regression (compiled)")
    except Exception as e:
        log(f"A.4 compile failed: {e}")
        results["regression_compiled"] = {"hz": 0, "error": str(e)}
    
    return results


# =============================================================================
# Main
# =============================================================================

def main():
    log("=" * 60)
    log("VLA-0 Fast Inference Pipeline")
    log("Target: 4 Hz (250ms)")
    log("=" * 60)
    
    # Load model
    model, cfg = load_model()
    rgb, instr = get_dummy_inputs()
    
    results = {}
    
    # Approach C: Compile + Static Cache
    try:
        results['approach_c'] = approach_c_compile_static_cache(model, rgb, instr)
    except Exception as e:
        log(f"Approach C failed: {e}")
        import traceback
        traceback.print_exc()
        results['approach_c'] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Approach D: Reduced Tokens
    try:
        results['approach_d'] = approach_d_reduced_tokens(model, rgb, instr)
    except Exception as e:
        log(f"Approach D failed: {e}")
        import traceback
        traceback.print_exc()
        results['approach_d'] = {"error": str(e)}
    
    gc.collect()
    torch.cuda.empty_cache()
    
    # Approach A: Regression Head
    try:
        results['approach_a'] = approach_a_regression_head(model, rgb, instr)
    except Exception as e:
        log(f"Approach A failed: {e}")
        import traceback
        traceback.print_exc()
        results['approach_a'] = {"error": str(e)}
    
    # Summary
    log("\n" + "=" * 60)
    log("SUMMARY")
    log("=" * 60)
    
    best_hz = 0
    best_approach = ""
    
    for approach, data in results.items():
        if isinstance(data, dict):
            for variant, metrics in data.items():
                if isinstance(metrics, dict) and 'hz' in metrics:
                    hz = metrics['hz']
                    ms = metrics.get('mean_ms', 0)
                    log(f"{approach}/{variant}: {hz:.2f} Hz ({ms:.1f} ms)")
                    if hz > best_hz:
                        best_hz = hz
                        best_approach = f"{approach}/{variant}"
    
    log(f"\nBest: {best_approach} at {best_hz:.2f} Hz")
    log(f"Target: 4.0 Hz (250ms)")
    log(f"Gap: {4.0 - best_hz:.2f} Hz")
    
    # Save results
    results_path = RESULTS_DIR / 'fast_inference_results.json'
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
