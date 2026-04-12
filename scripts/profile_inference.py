#!/usr/bin/env python3
"""
Profile VLA-0 inference to find where time is spent.
Break down: image preprocessing, tokenization, prefill, decode, action parsing.
"""
import os, sys, time, torch
import numpy as np

sys.path.insert(0, '/home/shadeform/vla0')
sys.path.insert(0, '/home/shadeform/vla0-compression')
os.chdir('/home/shadeform/vla0')

try:
    import roboverse.datasets.lerobot.dataloader as _rvlr
    class _MockMetadata:
        camera_keys = ['image', 'wrist_image']
    _rvlr.get_lerobot_metadata = lambda repo_id: _MockMetadata()
except:
    pass

CKPT = '/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth'

def log(msg):
    print(f"[PROFILE] {msg}", flush=True)

log("Loading model...")
from rv_train.train import get_pretrained_model
result = get_pretrained_model(CKPT, device='cuda')
model = result[0] if isinstance(result, tuple) else result
model.eval()

# Dummy input
rgb = torch.randint(0, 255, (1, 1, 2, 224, 224, 3), dtype=torch.uint8).float().cuda()
instruction = "put both the alphabet soup and the tomato sauce in the basket"

log("\n=== Warmup ===")
with torch.no_grad():
    for _ in range(3):
        model(rgb=rgb, instr=[instruction], get_action=True, get_loss=False)

log("\n=== Profiling forward pass ===")
torch.cuda.synchronize()

# Profile the full forward pass
times = []
for i in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(rgb=rgb, instr=[instruction], get_action=True, get_loss=False)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    times.append(t1 - t0)
    
log(f"Full forward: {np.mean(times)*1000:.0f}ms mean, {np.std(times)*1000:.0f}ms std")
log(f"  = {1/np.mean(times):.2f} Hz")

# Now profile individual components
log("\n=== Profiling components ===")

# 1. Image preprocessing (get_imgs)
img_times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    imgs = model.get_imgs(bs=1, pc=None, rgb_pc=None, rgb=rgb)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    img_times.append(t1 - t0)
log(f"1. get_imgs (tiling): {np.mean(img_times)*1000:.1f}ms")

# 2. Tokenization (get_qwen_inputs)
tok_times = []
for _ in range(10):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    model_inputs, examples = model.get_qwen_inputs(
        bs=1, imgs=imgs, instr=[instruction],
        action_txt=[[]], drop_assistant=True, add_generation_prompt=True
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    tok_times.append(t1 - t0)
log(f"2. get_qwen_inputs (tokenize+process): {np.mean(tok_times)*1000:.1f}ms")
log(f"   input_ids shape: {model_inputs['input_ids'].shape}")
n_tokens = model_inputs['input_ids'].shape[1]
log(f"   total input tokens: {n_tokens}")

# 3. Generation (model.generate)
gen_times = []
gen_tokens = []
for _ in range(5):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    generated_ids = model.model.generate(
        **model_inputs,
        max_new_tokens=280,  # 8*7*5 generous
        do_sample=False,
    )
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    gen_times.append(t1 - t0)
    new_tokens = generated_ids.shape[1] - n_tokens
    gen_tokens.append(new_tokens)
log(f"3. model.generate: {np.mean(gen_times)*1000:.0f}ms mean")
log(f"   generated tokens: {int(np.mean(gen_tokens))}")
log(f"   prefill throughput: {n_tokens/(np.mean(gen_times)):.0f} tok/s (input)")
log(f"   decode throughput: {np.mean(gen_tokens)/np.mean(gen_times):.0f} tok/s (output)")
log(f"   ms per output token: {np.mean(gen_times)/np.mean(gen_tokens)*1000:.1f}ms")

# 4. Action parsing
parse_times = []
text = model.processor.batch_decode(generated_ids[:, n_tokens:], skip_special_tokens=True)[0]
log(f"   generated text: {text[:200]}")
for _ in range(100):
    t0 = time.perf_counter()
    _ = model.get_action_from_text_action([text])
    parse_times.append(time.perf_counter() - t0)
log(f"4. Action parsing: {np.mean(parse_times)*1000:.2f}ms")

# Summary
total = np.mean(img_times) + np.mean(tok_times) + np.mean(gen_times) + np.mean(parse_times)
log(f"\n=== BREAKDOWN ===")
log(f"  Image prep:   {np.mean(img_times)/total*100:5.1f}%  ({np.mean(img_times)*1000:.0f}ms)")
log(f"  Tokenization: {np.mean(tok_times)/total*100:5.1f}%  ({np.mean(tok_times)*1000:.0f}ms)")
log(f"  Generation:   {np.mean(gen_times)/total*100:5.1f}%  ({np.mean(gen_times)*1000:.0f}ms)")
log(f"  Parsing:      {np.mean(parse_times)/total*100:5.1f}%  ({np.mean(parse_times)*1000:.2f}ms)")
log(f"  TOTAL:        {total*1000:.0f}ms = {1/total:.2f} Hz")

# Key insight: how many tokens does VLA-0 generate?
# 8 timesteps × 7 dims = 56 numbers. Each is 1-4 tokens. 
# Plus spaces. ~200-280 tokens total.
# At ~X ms/token decode, that's the bottleneck.

log(f"\n=== SPEED OPTIMIZATION TARGETS ===")
gen_ms = np.mean(gen_times)*1000
n_out = int(np.mean(gen_tokens))
ms_per_tok = gen_ms / n_out
log(f"Generation is {np.mean(gen_times)/total*100:.0f}% of total time")
log(f"Output tokens: {n_out}, at {ms_per_tok:.1f}ms/tok")
log(f"To reach 4 Hz (250ms budget):")
log(f"  Need {250/ms_per_tok:.0f} tok budget → reduce output tokens or speed up decode")
log(f"  Current: {n_out} output tokens")
log(f"  Options:")
log(f"    1. Reduce horizon (8→4): ~{n_out//2} tokens → ~{n_out//2*ms_per_tok:.0f}ms")
log(f"    2. Reduce bins (1000→256): fewer digits per number")
log(f"    3. Binary action encoding instead of text")
log(f"    4. torch.compile the generate loop")
log(f"    5. Static KV cache + CUDA graphs")
log(f"    6. TensorRT for the backbone")
log(f"    7. Speculative decoding")
