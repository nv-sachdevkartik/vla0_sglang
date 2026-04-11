"""
Utility to load VLA-0 model, bypassing the from_pretrained path.
Instead: create model from config, load state_dict from .pth checkpoint.
"""
import torch
torch.backends.cudnn.enabled = False

import sys, os, gc, pickle
sys.path.insert(0, '/home/shadeform/vla0')

from rv_train.train import get_model, get_cfg


def load_vla0_model(
    checkpoint_path='/home/shadeform/vla0-compression/checkpoints/vla0-original/model_last.pth',
    device=0,
):
    """Load VLA-0 model from checkpoint, manually loading state dict."""
    model_folder = os.path.dirname(checkpoint_path)
    cfg_path = os.path.join(model_folder, 'config.yaml')
    cfg = get_cfg(cfg_path, cfg_opts="")
    
    # Create model (this downloads Qwen2.5-VL-3B from HF)
    print("Creating model from config...")
    model = get_model(cfg, calculate_dataset_stats=False)
    model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Load state dict
    state_dict = checkpoint['model_state']
    # Handle DDP prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[len('module.'):]
        cleaned[k] = v
    
    result = model.load_state_dict(cleaned, strict=False)
    print(f"State dict loaded. Missing: {len(result.missing_keys)}, Unexpected: {len(result.unexpected_keys)}")
    if result.missing_keys:
        print(f"  Missing keys (first 5): {result.missing_keys[:5]}")
    if result.unexpected_keys:
        print(f"  Unexpected keys (first 5): {result.unexpected_keys[:5]}")
    
    # Load dataset stats
    stats_path = os.path.join(model_folder, 'dataset_stats.pkl')
    if os.path.exists(stats_path):
        with open(stats_path, 'rb') as f:
            dataset_stats = pickle.load(f)
        model.set_dataset_stats(dataset_stats)
        print("Dataset stats loaded.")
    
    # Cleanup
    del checkpoint, state_dict, cleaned
    torch.cuda.empty_cache()
    gc.collect()
    
    model.eval()
    print("Model loaded and set to eval mode!")
    print(f"  Children: {[n for n, _ in model.named_children()]}")
    print(f"  Params: {sum(p.numel() for p in model.parameters()) / 1e9:.3f} B")
    print(f"  Model size: {sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6:.1f} MB")
    
    return model, cfg


if __name__ == '__main__':
    model, cfg = load_vla0_model()
