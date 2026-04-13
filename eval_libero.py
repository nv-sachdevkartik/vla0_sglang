#!/usr/bin/env python3
"""
eval_libero_clean.py — LIBERO evaluation for VLA-0 via SGLang server.

Evaluates a VLA-0 vision-language-action model served by SGLang on the
LIBERO robot manipulation benchmark. Communicates with the SGLang server
over its OpenAI-compatible chat/completions API.

This script produced the validated 84.0% accuracy result on libero_10
(10 tasks × 5 seeds, action_horizon=8).

Prerequisites:
  1. SGLang server running and serving the VLA-0 model:
       python -m sglang.launch_server --model <model_dir> --port 30000 ...
  2. LIBERO environment (RoboVerse) installed in the current Python env.
  3. Dataset stats file at <checkpoint_dir>/dataset_stats.pkl.

Usage:
  python eval_libero_clean.py --server-url http://localhost:30000 --tasks 10 --seeds 5
  python eval_libero_clean.py --server-url http://localhost:30000 --horizon 8
"""

# ── EGL rendering setup (must happen before any MuJoCo / OpenGL imports) ─────
import os

os.environ["MUJOCO_GL"] = "egl"
os.environ["MUJOCO_EGL_DEVICE_ID"] = "1"
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["DISPLAY"] = ""
os.environ.pop("CUDA_VISIBLE_DEVICES", None)

# ── Standard library ─────────────────────────────────────────────────────────
import sys
import json
import time
import pickle
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime

# ── Path setup for VLA-0 and RoboVerse ───────────────────────────────────────
# Add VLA-0 source to path (needed for roboverse eval imports)
# Adjust these paths to your VLA-0 installation
VLA0_ROOT = os.environ.get("VLA0_ROOT", "/home/shadeform/vla0")
sys.path.insert(0, VLA0_ROOT)
os.chdir(VLA0_ROOT)

# ── LeRobot metadata monkey-patch ────────────────────────────────────────────
# WHY: The LIBERO eval pipeline imports roboverse's lerobot dataloader, which
# calls `get_lerobot_metadata(repo_id)` to discover camera keys for a dataset.
# In our offline eval setting there is no HuggingFace Hub dataset repo to query,
# so the call would fail. We patch it to return a stub with the two camera keys
# VLA-0 expects ("image" for 3rd-person, "wrist_image" for wrist camera).
# This avoids a network dependency during evaluation without changing any
# evaluation logic.
try:
    import roboverse.datasets.lerobot.dataloader as _rv_lerobot

    class _StubMetadata:
        camera_keys = ["image", "wrist_image"]

    _rv_lerobot.get_lerobot_metadata = lambda repo_id: _StubMetadata()
except Exception:
    pass

# ── Constants ────────────────────────────────────────────────────────────────
DEFAULT_MODEL_DIR = (
    "./model"
)
DEFAULT_STATS_PATH = (
    "./dataset_stats.pkl"
)
# Resolve to absolute path before os.chdir changes cwd
RESULTS_BASE = Path(os.path.abspath("./results/eval"))

# The exact system message VLA-0 was fine-tuned with.
SYSTEM_MESSAGE_TEMPLATE = (
    "Analyze the input image and predict robot actions for the next {horizon} timesteps. "
    "Each action has {act_dim} dimensions. Output a single sequence of {total} integers "
    "(0-{num_bins} each), representing the {horizon} timesteps sequentially. "
    "Provide only space separated numbers. Nothing else."
)

NUM_BINS = 1000
ACT_DIM = 7


# ── Helpers ──────────────────────────────────────────────────────────────────


def log(msg: str) -> None:
    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_dataset_stats(stats_path: str = DEFAULT_STATS_PATH) -> dict:
    with open(stats_path, "rb") as f:
        stats = pickle.load(f)
    return stats.get("out_ori_act", stats)


# ── SGLang client ────────────────────────────────────────────────────────────

import base64
import io

import requests
import torch
from PIL import Image


class SGLangClient:
    """Drop-in replacement for QwenActor inference, backed by an SGLang server.

    Communicates via the OpenAI-compatible ``/v1/chat/completions`` endpoint.
    Uses ``requests.Session`` for TCP connection reuse (avoids connection leaks
    that occurred with per-call ``requests.post``).

    The prompt format exactly matches VLA-0's training:
      - system: "Analyze the input image …"
      - user:   [tiled camera image] + task instruction text
      - assistant: space-separated integer tokens → denormalized to actions
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        horizon: int,
        dataset_stats: dict,
        num_bins: int = NUM_BINS,
        act_dim: int = ACT_DIM,
    ):
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.num_bins = num_bins
        self.act_dim = act_dim
        self.horizon = horizon
        self.url = f"{self.base_url}/v1/chat/completions"

        self.system_message = SYSTEM_MESSAGE_TEMPLATE.format(
            horizon=horizon,
            act_dim=act_dim,
            total=horizon * act_dim,
            num_bins=num_bins,
        )

        # Denormalization stats
        self.min_act = np.array(dataset_stats["min"], dtype=np.float32)
        self.max_act = np.array(dataset_stats["max"], dtype=np.float32)

        # Persistent HTTP session with connection pooling and retries
        self._session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=requests.adapters.Retry(total=3, backoff_factor=0.5),
        )
        self._session.mount("http://", adapter)

        # Latency tracking
        self.call_count = 0
        self.total_latency_ms = 0.0

    # ── Image encoding ───────────────────────────────────────────────────

    @staticmethod
    def _rgb_to_tiled_pil(rgb_tensor) -> Image.Image:
        """Convert VLA-0 RGB tensor to a single horizontally-tiled PIL image.

        Input shape: ``[B, history, num_cam, H, W, C]`` float32 (0-255).
        VLA-0 was fine-tuned with ``tiled_rgb_imgs=True``, so the two cameras
        (3rd-person + wrist) are tiled side-by-side into one 224×448 image.
        """
        rgb = rgb_tensor[0, -1]  # [num_cam, H, W, C] — latest frame
        frames = [rgb[i].cpu().numpy().astype(np.uint8) for i in range(rgb.shape[0])]
        tiled = np.concatenate(frames, axis=1)  # [H, num_cam*W, C]
        return Image.fromarray(tiled)

    @staticmethod
    def _encode_image(pil_img: Image.Image) -> str:
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    # ── Action parsing ───────────────────────────────────────────────────

    def _parse_action_text(self, text: str):
        """Parse space-separated integer tokens into a denormalized action array.

        Reproduces QwenActor.get_action_from_text_action exactly:
          1. Split text, keep valid integers
          2. Truncate to a multiple of ``act_dim``
          3. Reshape → ``(steps, act_dim)``, pad/trim to ``horizon``
          4. Denormalize: ``action = (binned / num_bins) * (max - min) + min``
        """
        try:
            numbers = []
            for tok in text.strip().split():
                try:
                    numbers.append(int(tok))
                except ValueError:
                    continue
            if not numbers:
                return None

            n = len(numbers) - (len(numbers) % self.act_dim)
            if n == 0:
                return None
            numbers = numbers[:n]

            actions = np.array(numbers, dtype=np.float32).reshape(-1, self.act_dim)

            # Pad with last action if fewer than horizon steps
            if len(actions) < self.horizon:
                pad = np.tile(actions[-1:], (self.horizon - len(actions), 1))
                actions = np.concatenate([actions, pad])
            actions = actions[: self.horizon]

            # Denormalize
            actions = (
                (actions / self.num_bins) * (self.max_act - self.min_act)
            ) + self.min_act
            return actions

        except Exception as e:
            log(f"[SGLangClient] Action parse error: {e}, text: {text[:200]}")
            return None

    # ── Main call (QwenActor.forward interface) ──────────────────────────

    def __call__(self, rgb=None, instr=None, get_action=True, get_loss=False, **kwargs):
        """Match the ``QwenActor.forward()`` interface expected by LIBERO eval."""
        if not get_action:
            return {}

        tiled_img = self._rgb_to_tiled_pil(rgb)
        instruction = instr[0] if isinstance(instr, list) else instr

        messages = [
            {"role": "system", "content": self.system_message},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": self._encode_image(tiled_img)},
                    },
                    {"type": "text", "text": instruction},
                ],
            },
        ]

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.horizon * self.act_dim * 5,
            "temperature": 0.0,
        }

        t0 = time.perf_counter()
        try:
            response = self._session.post(self.url, json=payload, timeout=120)
            latency_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            log(f"[SGLangClient] Request error: {e}")
            return {"out_ori_act": torch.zeros(1, self.horizon, self.act_dim)}

        if response.status_code != 200:
            log(
                f"[SGLangClient] API error: {response.status_code} "
                f"{response.text[:300]}"
            )
            return {"out_ori_act": torch.zeros(1, self.horizon, self.act_dim)}

        self.call_count += 1
        self.total_latency_ms += latency_ms
        if self.call_count % 50 == 0:
            avg = self.total_latency_ms / self.call_count
            log(f"    [HTTP] {self.call_count} calls, avg {avg:.0f}ms/call")

        result = response.json()
        action_text = result["choices"][0]["message"]["content"]

        actions = self._parse_action_text(action_text)
        if actions is None:
            log(f"[SGLangClient] Failed to parse: {action_text[:200]}")
            mid = (self.min_act + self.max_act) / 2
            out = (
                torch.tensor(mid, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(1, self.horizon, 1)
            )
            return {"out_ori_act": out}

        out = torch.tensor(actions, dtype=torch.float32).unsqueeze(0)  # [1, H, act_dim]
        return {
            "out_ori_act": out,
            "pred_action_txt": [action_text],
            "latency_ms": latency_ms,
        }

    def close(self):
        self._session.close()


# ── LIBERO eval loop ─────────────────────────────────────────────────────────


def run_full_eval(
    model,
    label: str,
    action_horizon: int = 8,
    task_suite: str = "libero_10",
    num_tasks: int = 10,
    num_seeds: int = 5,
) -> dict:
    """Run the full LIBERO evaluation: ``num_tasks × num_seeds`` episodes.

    Each task is evaluated via ``roboverse.evals.libero.eval.eval``, which
    runs ``num_seeds`` rollouts per task and writes per-task ``results.json``.
    """
    from roboverse.evals.libero.eval import eval as libero_eval, get_evaluation_tasks

    tasks_dict = get_evaluation_tasks(task_suite_name=task_suite)
    task_names = tasks_dict[task_suite][:num_tasks]

    total_episodes = len(task_names) * num_seeds
    log(
        f"LIBERO eval [{label}] — {len(task_names)} tasks × {num_seeds} seeds "
        f"= {total_episodes} episodes"
    )
    log(f"  action_horizon={action_horizon}")

    eval_dir = RESULTS_BASE / label
    eval_dir.mkdir(parents=True, exist_ok=True)

    cfg_path = "libs/RoboVerse/roboverse/configs/img_libero_aug.yaml"
    cfg_opts = "IMAGE.crop_img:0.875:IMAGE.img_size:224:IMAGE.cam_list:('3p1','3p2')"

    results = {
        "label": label,
        "action_horizon": action_horizon,
        "task_suite": task_suite,
        "num_tasks": len(task_names),
        "num_seeds": num_seeds,
        "tasks": {},
        "total_success": 0,
        "total_trials": 0,
    }

    episode_num = 0
    eval_start = time.perf_counter()

    for i, task_name in enumerate(task_names):
        short_name = task_name.split("_", 3)[-1][:50]
        log(f"  [{i + 1}/{len(task_names)}] {short_name}...")
        task_dir = eval_dir / task_suite / task_name
        task_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.perf_counter()
        try:
            libero_eval(
                model=model,
                action_type="original",
                cfg_path=cfg_path,
                cfg_opts=cfg_opts,
                task_name=task_name,
                task_suite_name=task_suite,
                log_dir=str(task_dir),
                save_video=True,
                seed=7,
                action_horizon=action_horizon,
                skip_evaluated=False,
                task_id_index=0,
                task_id_count=num_tasks,
                num_steps=0,
            )

            rf = task_dir / "results.json"
            if rf.exists():
                with open(rf) as f:
                    tr = json.load(f)
                s = tr.get("success", 0)
                fail = tr.get("failure", 0)
                total = s + fail
                rate = s / total if total else 0
                results["tasks"][task_name] = {
                    "success": s,
                    "failure": fail,
                    "rate": rate,
                    "time_seconds": time.perf_counter() - t0,
                }
                results["total_success"] += s
                results["total_trials"] += total
                episode_num += total

                elapsed = time.perf_counter() - t0
                overall_elapsed = time.perf_counter() - eval_start
                eta_per_ep = overall_elapsed / episode_num if episode_num else 0
                remaining = (total_episodes - episode_num) * eta_per_ep

                log(
                    f"    → {s}/{total} ({100 * rate:.0f}%) in {elapsed:.0f}s | "
                    f"{episode_num}/{total_episodes} done | ETA {remaining / 60:.0f}min"
                )
        except Exception as e:
            log(f"    → ERROR: {e}")
            import traceback

            traceback.print_exc()
            results["tasks"][task_name] = {"error": str(e)}

    total_time = time.perf_counter() - eval_start

    if results["total_trials"] > 0:
        results["success_rate"] = results["total_success"] / results["total_trials"]
    results["total_time_seconds"] = total_time
    results["time_per_episode"] = (
        total_time / results["total_trials"] if results["total_trials"] else 0
    )

    with open(eval_dir / "summary.json", "w") as f:
        json.dump(results, f, indent=2)

    log(
        f'\n  FINAL: {results["total_success"]}/{results["total_trials"]} = '
        f'{100 * results.get("success_rate", 0):.1f}% in {total_time / 60:.1f}min'
    )

    return results


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate VLA-0 on LIBERO via SGLang server.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--server-url",
        default="http://localhost:30000",
        help="Base URL of the SGLang server.",
    )
    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_DIR,
        help="Model name/path to pass in API requests.",
    )
    parser.add_argument(
        "--tasks", type=int, default=10, help="Number of LIBERO tasks to evaluate."
    )
    parser.add_argument(
        "--seeds", type=int, default=5, help="Number of seeds (rollouts) per task."
    )
    parser.add_argument(
        "--horizon", type=int, default=8, help="Action prediction horizon (timesteps)."
    )
    parser.add_argument(
        "--stats-path",
        default=DEFAULT_STATS_PATH,
        help="Path to dataset_stats.pkl for action denormalization.",
    )
    args = parser.parse_args()

    # Resolve all paths to absolute BEFORE os.chdir changes cwd
    args.stats_path = os.path.abspath(args.stats_path)
    args.model_name = os.path.abspath(args.model_name) if not args.model_name.startswith("http") else args.model_name

    RESULTS_BASE.mkdir(parents=True, exist_ok=True)
    stats = load_dataset_stats(args.stats_path)

    # ── Health check ─────────────────────────────────────────────────────
    log(f"Checking SGLang server at {args.server_url} ...")
    try:
        r = requests.get(f"{args.server_url.rstrip('/')}/health", timeout=10)
        log(f"Server healthy: {r.status_code}")
    except Exception as e:
        log(f"FATAL: SGLang server not reachable at {args.server_url}: {e}")
        sys.exit(1)

    # ── Build client and run ─────────────────────────────────────────────
    client = SGLangClient(
        base_url=args.server_url,
        model_name=args.model_name,
        horizon=args.horizon,
        dataset_stats=stats,
    )

    label = f"sglang_bf16_{args.horizon}step"
    try:
        run_full_eval(
            client,
            label,
            action_horizon=args.horizon,
            num_tasks=args.tasks,
            num_seeds=args.seeds,
        )
    finally:
        client.close()

    log("=== DONE ===")


if __name__ == "__main__":
    main()
