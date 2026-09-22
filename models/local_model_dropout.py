import torch
import torch.nn as nn
import torch.nn.functional as F


class ShapeNetPointsDropout(nn.Module):
    """IF-Net with scale dropout regularisation.

    Architecture is identical to ShapeNetPoints (baseline).
    The only difference is _scale_dropout() applied to each
    feature scale during training, randomly zeroing entire
    scale levels with probability `scale_dropout_p`.

    During inference (self.training == False), no dropout is
    applied — behaviour is identical to baseline IF-Net.
    """

    def __init__(self, hidden_dim=256, scale_dropout_p=0.1):
        super(ShapeNetPointsDropout, self).__init__()

        self.scale_dropout_p = scale_dropout_p

        # ── Encoder: identical to ShapeNetPoints ──────────────
        # 128³ input → conv_in → BN → maxpool → 64³
        self.conv_in   = nn.Conv3d(1,   16,  3, padding=1, padding_mode='replicate')
        # 64³ → conv_0 → conv_0_1 → BN → maxpool → 32³
        self.conv_0    = nn.Conv3d(16,  32,  3, padding=1, padding_mode='replicate')
        self.conv_0_1  = nn.Conv3d(32,  32,  3, padding=1, padding_mode='replicate')
        # 32³ → conv_1 → conv_1_1 → BN → maxpool → 16³
        self.conv_1    = nn.Conv3d(32,  64,  3, padding=1, padding_mode='replicate')
        self.conv_1_1  = nn.Conv3d(64,  64,  3, padding=1, padding_mode='replicate')
        # 16³ → conv_2 → conv_2_1 → BN → maxpool → 8³
        self.conv_2    = nn.Conv3d(64,  128, 3, padding=1, padding_mode='replicate')
        self.conv_2_1  = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')
        # 8³ → conv_3 → conv_3_1 → BN
        self.conv_3    = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')
        self.conv_3_1  = nn.Conv3d(128, 128, 3, padding=1, padding_mode='replicate')

        # ── Decoder: identical to ShapeNetPoints ──────────────
        # 6 scales: ch = 1 + 16 + 32 + 64 + 128 + 128 = 369
        # 7 displacement samples → feature_size = 369 * 7 = 2583
        feature_size = (1 + 16 + 32 + 64 + 128 + 128) * 7
        self.fc_0   = nn.Conv1d(feature_size, hidden_dim, 1)
        self.fc_1   = nn.Conv1d(hidden_dim,   hidden_dim, 1)
        self.fc_2   = nn.Conv1d(hidden_dim,   hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim,   1,          1)
        self.actvn  = nn.ReLU()

        # ── Pooling & BatchNorm: identical to ShapeNetPoints ──
        self.maxpool     = nn.MaxPool3d(2)
        self.conv_in_bn  = nn.BatchNorm3d(16)
        self.conv0_1_bn  = nn.BatchNorm3d(32)
        self.conv1_1_bn  = nn.BatchNorm3d(64)
        self.conv2_1_bn  = nn.BatchNorm3d(128)
        self.conv3_1_bn  = nn.BatchNorm3d(128)

        # ── Displacement offsets: identical to ShapeNetPoints ──
        displacment = 0.0722
        displacments = [[0, 0, 0]]
        for x in range(3):
            for y in [-1, 1]:
                d = [0, 0, 0]
                d[x] = y * displacment
                displacments.append(d)
        self.displacments = torch.Tensor(displacments).cuda()

    def _scale_dropout(self, feature):
        """Zero out entire scale during training with prob scale_dropout_p.

        A single Bernoulli draw per batch item decides whether the
        entire scale level is kept (1) or zeroed (0).  At inference
        time this is a no-op.
        """
        if self.training and self.scale_dropout_p > 0:
            mask = torch.bernoulli(
                torch.ones(feature.shape[0], 1, 1, 1, 1,
                           device=feature.device)
                * (1.0 - self.scale_dropout_p)
            )
            feature = feature * mask
        return feature

    def forward(self, p, x):
        x = x.unsqueeze(1)

        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)

        # ── Scale 0: raw input (ch=1, 128³) ───────────────────
        feature_0 = F.grid_sample(x, p, padding_mode='border')

        # ── Scale 1: after conv_in (ch=16, 128³) ──────────────
        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)                                  # → 64³

        # ── Scale 2: after conv_0_1 (ch=32, 64³) ──────────────
        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)                                  # → 32³

        # ── Scale 3: after conv_1_1 (ch=64, 32³) ──────────────
        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)                                  # → 16³

        # ── Scale 4: after conv_2_1 (ch=128, 16³) ─────────────
        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode='border')
        net = self.maxpool(net)                                  # → 8³

        # ── Scale 5: after conv_3_1 (ch=128, 8³) ──────────────
        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode='border')

        # ── Apply scale dropout independently to each scale ────
        feature_0 = self._scale_dropout(feature_0)
        feature_1 = self._scale_dropout(feature_1)
        feature_2 = self._scale_dropout(feature_2)
        feature_3 = self._scale_dropout(feature_3)
        feature_4 = self._scale_dropout(feature_4)
        feature_5 = self._scale_dropout(feature_5)

        # ── Concatenate & decode ───────────────────────────────
        features = torch.cat(
            (feature_0, feature_1, feature_2, feature_3, feature_4, feature_5),
            dim=1
        )
        shape = features.shape
        features = torch.reshape(
            features, (shape[0], shape[1] * shape[3], shape[4])
        )

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        net = self.fc_out(net)
        out = net.squeeze(1)

        return out