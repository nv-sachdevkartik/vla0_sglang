#!/usr/bin/env python3
"""
VLA-0 SGLang Inference Client

Production client for VLA-0 robot policy inference via SGLang server.

Validated: 84.0% accuracy on LIBERO-10 (identical to PyTorch baseline)
Speed: 4.8 Hz one-step / 0.93 Hz eight-step on H100 PCIe

Requirements:
    numpy
    requests
    Pillow

Usage::

    client = VLA0Client(
        server_url="http://localhost:30000",
        model_path="your/model/path",
    )
    action = client.predict(rgb_image, "pick up the red block")
    # action.shape = (1, 7) — 7-DOF robot action per horizon step

Note:
    The SGLang server must be started separately before using this client.
    Example server launch::

        python -m sglang.launch_server \\
            --model-path your/model/path \\
            --port 30000 \\
            --trust-remote-code \\
            --mem-fraction-static 0.15 \\
            --max-total-tokens 512 \\
            --max-running-requests 1 \\
            --dtype auto
"""

from __future__ import annotations

import base64
import io
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import requests
from PIL import Image

__all__ = [
    "VLA0Client",
    "VLA0Config",
    "DEFAULT_STATS",
    "SYSTEM_MESSAGE",
]


# ─── Configuration ───────────────────────────────────────────────────────────

SYSTEM_MESSAGE: str = (
    "Analyze the input image and predict robot actions for the next {horizon} timesteps. "
    "Each action has {act_dim} dimensions. Output a single sequence of {total} integers "
    "(0-{num_bins} each), representing the {horizon} timesteps sequentially. "
    "Provide only space separated numbers. Nothing else."
)

# Default dataset stats from VLA-0 LIBERO checkpoint (dataset_stats.pkl).
# Used to denormalize actions from [0, num_bins] → physical space.
DEFAULT_STATS: Dict[str, List[float]] = {
    "min": [-0.9375, -0.9375, -0.9375, -0.2582, -0.375, -0.3675, -1.0],
    "max": [ 0.9375,  0.9375,  0.9375,  0.3557,  0.375,  0.375,   1.0],
}


@dataclass
class VLA0Config:
    """VLA-0 model configuration.

    Attributes:
        num_bins: Number of action discretization bins.
        act_dim: Action dimensionality (6-DOF + gripper).
        horizon: Action prediction horizon (1 for speed, 8 for quality).
        server_url: Base URL of the SGLang server.
        model_path: HuggingFace model path as loaded by SGLang.
    """

    num_bins: int = 1000
    act_dim: int = 7
    horizon: int = 8
    server_url: str = "http://localhost:30000"
    model_path: str = ""


class VLA0Client:
    """
    Production VLA-0 client for SGLang serving.

    Sends RGB observations and language instructions to an SGLang server
    running a VLA-0 checkpoint and returns denormalized robot actions.

    Args:
        server_url: SGLang server URL.
        model_path: HuggingFace model path as loaded by SGLang.
        horizon: Action prediction horizon. Use 1 for maximum speed (4.8 Hz),
                 8 for maximum action quality (0.93 Hz).
        dataset_stats: Dict with ``"min"`` and ``"max"`` float lists for action
                       denormalization. Defaults to :data:`DEFAULT_STATS`.
    """

    def __init__(
        self,
        server_url: str = "http://localhost:30000",
        model_path: str = "",
        horizon: int = 1,
        dataset_stats: Optional[Dict[str, List[float]]] = None,
    ) -> None:
        if not model_path:
            raise ValueError(
                "model_path is required (e.g. 'your-org/vla0-checkpoint'). "
                "Pass the same path used when launching the SGLang server."
            )

        self.config: VLA0Config = VLA0Config(
            server_url=server_url,
            model_path=model_path,
            horizon=horizon,
        )

        # ── HTTP session with connection pooling ──────────────────────────
        # WARNING: Using requests.Session() for connection reuse is CRITICAL.
        # Without it, each request opens a new TCP connection. SGLang does not
        # always close server-side sockets promptly, which causes the server to
        # accumulate stale CLOSE_WAIT / TIME_WAIT connections and eventually
        # become unresponsive. A single persistent session avoids this entirely.
        self._session: requests.Session = requests.Session()
        adapter: requests.adapters.HTTPAdapter = requests.adapters.HTTPAdapter(
            pool_connections=1,
            pool_maxsize=1,
            max_retries=requests.adapters.Retry(total=3, backoff_factor=0.3),
        )
        self._session.mount("http://", adapter)
        self._session.mount("https://", adapter)

        # System message (must exactly match training format)
        self._system_msg: str = SYSTEM_MESSAGE.format(
            horizon=self.config.horizon,
            act_dim=self.config.act_dim,
            total=self.config.horizon * self.config.act_dim,
            num_bins=self.config.num_bins,
        )

        # Action denormalization stats
        stats: Dict[str, List[float]] = dataset_stats or DEFAULT_STATS
        self._min_act: np.ndarray = np.array(stats["min"], dtype=np.float32)
        self._max_act: np.ndarray = np.array(stats["max"], dtype=np.float32)

        # Max tokens: each action value is up to 4 chars + space
        self._max_tokens: int = self.config.horizon * self.config.act_dim * 5

        # Latency tracking
        self._latencies: List[float] = []

    # ── Public API ────────────────────────────────────────────────────────

    def predict(
        self,
        rgb: np.ndarray,
        instruction: str,
        temperature: float = 0.0,
    ) -> np.ndarray:
        """
        Predict robot action from RGB observation and language instruction.

        Args:
            rgb: RGB image as a numpy array. Supported shapes:

                 * ``(H, W, 3)`` — single camera image.
                 * ``(H, W*2, 3)`` — pre-tiled dual camera image.
                 * ``(2, H, W, 3)`` — dual camera, tiled horizontally.
            instruction: Natural language task instruction.
            temperature: Sampling temperature (0.0 = greedy, recommended).

        Returns:
            ``np.ndarray`` of shape ``(horizon, 7)`` — denormalized robot
            actions. Each row is ``[dx, dy, dz, rx, ry, rz, gripper]``.
            For ``horizon=1``, returns shape ``(1, 7)``; use ``action[0]``
            for a single step.
        """
        image_b64: str = self._encode_image(rgb)

        payload: dict = {
            "model": self.config.model_path,
            "messages": [
                {"role": "system", "content": self._system_msg},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                        {"type": "text", "text": instruction},
                    ],
                },
            ],
            "max_tokens": self._max_tokens,
            "temperature": temperature,
        }

        t0: float = time.perf_counter()
        try:
            resp: requests.Response = self._session.post(
                f"{self.config.server_url}/v1/chat/completions",
                json=payload,
                timeout=30,
            )
            latency_ms: float = (time.perf_counter() - t0) * 1000
            self._latencies.append(latency_ms)
        except requests.RequestException as e:
            print(f"[VLA0Client] Request failed: {e}")
            return self._zero_action()

        if resp.status_code != 200:
            print(f"[VLA0Client] Server error {resp.status_code}: {resp.text[:200]}")
            return self._zero_action()

        text: str = resp.json()["choices"][0]["message"]["content"]
        actions: Optional[np.ndarray] = self._parse_actions(text)

        if actions is None:
            print(f"[VLA0Client] Failed to parse: {text[:200]}")
            return self._zero_action()

        return actions

    def predict_single(self, rgb: np.ndarray, instruction: str) -> np.ndarray:
        """Predict and return a single action step as a ``(7,)`` array."""
        return self.predict(rgb, instruction)[0]

    def health_check(self) -> bool:
        """Return ``True`` if the SGLang server is responsive."""
        try:
            r: requests.Response = self._session.get(
                f"{self.config.server_url}/v1/models", timeout=5,
            )
            return r.status_code == 200
        except Exception:
            return False

    @property
    def avg_latency_ms(self) -> float:
        """Average request latency in milliseconds."""
        return float(np.mean(self._latencies)) if self._latencies else 0.0

    def close(self) -> None:
        """Close the underlying HTTP session."""
        self._session.close()

    # ── Context manager support ───────────────────────────────────────────

    def __enter__(self) -> "VLA0Client":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self._session.close()
        except Exception:
            pass

    # ── Private helpers ───────────────────────────────────────────────────

    def _encode_image(self, rgb: np.ndarray) -> str:
        """Convert numpy RGB array to a base64-encoded PNG string."""
        if rgb.ndim == 4 and rgb.shape[0] == 2:
            # (2, H, W, 3) → tile dual cameras horizontally
            rgb = np.concatenate([rgb[0], rgb[1]], axis=1)

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        pil: Image.Image = Image.fromarray(rgb)
        buf: io.BytesIO = io.BytesIO()
        pil.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode()

    def _parse_actions(self, text: str) -> Optional[np.ndarray]:
        """Parse space-separated integers into denormalized action array."""
        try:
            tokens: List[str] = text.strip().split()
            numbers: List[int] = [
                int(t) for t in tokens
                if t.isdigit() or (t.startswith("-") and t[1:].isdigit())
            ]

            if not numbers:
                return None

            # Truncate to a multiple of act_dim
            n: int = len(numbers) - (len(numbers) % self.config.act_dim)
            if n == 0:
                return None

            actions: np.ndarray = (
                np.array(numbers[:n], dtype=np.float32)
                .reshape(-1, self.config.act_dim)
            )

            # Pad to horizon if the model returned fewer steps
            if len(actions) < self.config.horizon:
                pad: np.ndarray = np.tile(
                    actions[-1:], (self.config.horizon - len(actions), 1),
                )
                actions = np.concatenate([actions, pad])
            actions = actions[: self.config.horizon]

            # Denormalize: action = (binned / num_bins) * (max - min) + min
            actions = (
                (actions / self.config.num_bins)
                * (self._max_act - self._min_act)
                + self._min_act
            )

            return actions
        except Exception as e:
            print(f"[VLA0Client] Parse error: {e}")
            return None

    def _zero_action(self) -> np.ndarray:
        """Return midpoint actions as a safe fallback."""
        mid: np.ndarray = (self._min_act + self._max_act) / 2
        return np.tile(mid, (self.config.horizon, 1))


# ─── CLI Demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser: argparse.ArgumentParser = argparse.ArgumentParser(
        description="VLA-0 SGLang Inference Client — Demo",
    )
    parser.add_argument(
        "--url", default="http://localhost:30000", help="SGLang server URL",
    )
    parser.add_argument(
        "--model", required=True, help="Model path (as loaded by SGLang)",
    )
    parser.add_argument(
        "--horizon", type=int, default=1,
        help="Prediction horizon: 1 = fast (4.8 Hz), 8 = full quality",
    )
    args: argparse.Namespace = parser.parse_args()

    with VLA0Client(
        server_url=args.url,
        model_path=args.model,
        horizon=args.horizon,
    ) as client:
        if not client.health_check():
            print("ERROR: SGLang server not available at", args.url)
            print("Start it first, e.g.:")
            print(
                f"  python -m sglang.launch_server --model-path {args.model} "
                "--port 30000 --trust-remote-code --mem-fraction-static 0.15 "
                "--max-total-tokens 512"
            )
            raise SystemExit(1)

        # Demo with a random image
        dummy_rgb: np.ndarray = np.random.randint(
            0, 255, (224, 448, 3), dtype=np.uint8,
        )
        action: np.ndarray = client.predict(dummy_rgb, "pick up the red block")

        print(f"Action ({args.horizon}-step): shape={action.shape}")
        print(f"Step 0: {action[0]}")
        print(f"Latency: {client.avg_latency_ms:.0f}ms")
