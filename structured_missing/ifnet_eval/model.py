"""Checkpoint-compatible inference adapter for cccku/if-net-robustness.

Derived from upstream ShapeNetPoints/ShapeNetPointsDropout at
7056f2ead1e7d6ec306c8b9c5f870c45af34118e. Layer names, order,
offsets and sampling semantics are preserved. The encoder is cached.
In eval mode both architectures have the same forward computation;
their learned weights differ. No upstream training implementation is changed.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def world_to_grid(points: np.ndarray) -> np.ndarray:
    """World xyz in [-.5,.5] -> grid_sample zyx in [-1,1], ONCE."""
    return np.ascontiguousarray(np.asarray(points, dtype=np.float32)[:, [2, 1, 0]] * 2)


class CachedIFNet(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.conv_in = nn.Conv3d(1, 16, 3, padding=1, padding_mode="replicate")
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1, padding_mode="replicate")
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1, padding_mode="replicate")
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1, padding_mode="replicate")
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1, padding_mode="replicate")
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode="replicate")
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode="replicate")
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1, padding_mode="replicate")
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1, padding_mode="replicate")
        self.fc_0 = nn.Conv1d(2583, hidden_dim, 1)
        self.fc_1 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()
        self.maxpool = nn.MaxPool3d(2)
        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        offsets = [[0., 0., 0.]]
        for axis in range(3):
            for sign in (-1, 1):
                offset = [0., 0., 0.]
                offset[axis] = sign * 0.0722
                offsets.append(offset)
        # Upstream offsets are not in state_dict; preserve strict compatibility.
        self.register_buffer("displacments", torch.tensor(offsets), persistent=False)

    def encode(self, inputs: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if self.training:
            raise RuntimeError("This adapter is inference-only; call eval().")
        net = inputs.unsqueeze(1)
        features = [net]
        net = self.conv_in_bn(self.actvn(self.conv_in(net)))
        features.append(net)
        for first, second, bn in (
            (self.conv_0, self.conv_0_1, self.conv0_1_bn),
            (self.conv_1, self.conv_1_1, self.conv1_1_bn),
            (self.conv_2, self.conv_2_1, self.conv2_1_bn),
            (self.conv_3, self.conv_3_1, self.conv3_1_bn),
        ):
            net = self.maxpool(net)
            net = bn(self.actvn(second(self.actvn(first(net)))))
            features.append(net)
        return tuple(features)

    def decode(self, grid_coords: torch.Tensor, features, masked_scales=()):
        if any(i not in range(6) for i in masked_scales):
            raise ValueError("Scale IDs must be 0..5")
        p = grid_coords.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + offset for offset in self.displacments], dim=2)
        sampled = []
        for i, feature in enumerate(features):
            values = F.grid_sample(feature, p, mode="bilinear",
                                   padding_mode="border", align_corners=False)
            # Mask ONLY the decoder branch, never the encoder propagation.
            sampled.append(torch.zeros_like(values) if i in masked_scales else values)
        net = torch.cat(sampled, dim=1)
        b, c, _, offsets, n = net.shape
        net = net.reshape(b, c * offsets, n)
        net = self.actvn(self.fc_0(net))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        return self.fc_out(net).squeeze(1)

    def forward(self, p, inputs):
        return self.decode(p, self.encode(inputs))


def load_model(path: Path, device="cpu"):
    # No arbitrary pickle execution fallback. These two supplied files load safely.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state = checkpoint["model_state_dict"]
    net = CachedIFNet(hidden_dim=state["fc_0.weight"].shape[0])
    net.load_state_dict(state, strict=True)
    net.eval().to(device)
    metadata = {"checkpoint": str(Path(path).resolve()),
                "sha256": file_sha256(path), "epoch": int(checkpoint["epoch"]),
                "state_keys": len(state), "strict_load": True}
    return net, metadata


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def query_world(net, features, points, chunk_size=2048, masked_scales=()):
    device = features[0].device
    predictions = []
    with torch.inference_mode():
        for start in range(0, len(points), chunk_size):
            coords = torch.from_numpy(world_to_grid(points[start:start + chunk_size]))
            logits = net.decode(coords.unsqueeze(0).to(device), features, masked_scales)
            predictions.append(logits.squeeze(0).cpu().numpy())
    return np.concatenate(predictions)


def timed_encode(net, voxels, device):
    synchronize(device)
    start = time.perf_counter()
    with torch.inference_mode():
        features = net.encode(torch.from_numpy(voxels[None]).to(device))
    synchronize(device)
    return features, time.perf_counter() - start

