import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque  # For fixed-length history


class DynamicThresholdFocalLoss(nn.Module):
    """
    Optimized Adaptive Threshold Focal Loss (solving attribute definition, memory, and initial stability issues)
    """

    def __init__(self,
                 num_classes,
                 gamma=1.5,  # N:1.5; S:1.4; M:1.3; L:1.2; X:1.1
                 alpha=None,
                 window_size=50,  # N:50; S:60; M:70; L:80; X:100
                 temperature=0.8,  # N:0.8; S:0.85; M:0.9; L:0.92; X:0.95
                 label_smoothing=0.1,  # N:0.1; S:0.08; M:0.06; L:0.04; X:0.02
                 gamma_clamp=(0.5, 2.0)  # Explicitly pass the upper and lower limits of gamma (core modification: as constructor parameters)
                ):
        super(DynamicThresholdFocalLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = torch.tensor(gamma, dtype=torch.float32)  # Keep tensor type
        self.temperature = temperature
        self.window_size = window_size
        self.label_smoothing = label_smoothing
        # Fix 1: Replace the list with a deque, with a fixed length that is automatically maintained (to solve the memory growth problem)
        self.p_t_history = deque(maxlen=window_size)  # Automatically discard old values when exceeding window_size
        self.gamma_clamp = gamma_clamp  # Fix 2: Explicitly define as a class attribute (rather than adding it dynamically)

        # Class Balance Factor
        if alpha is None:
            self.alpha_pos = torch.ones(num_classes, dtype=torch.float32) * 0.7  # N Default
            self.alpha_neg = torch.ones(num_classes, dtype=torch.float32) * 0.3
        else:
            self.alpha_pos = torch.tensor(alpha[0], dtype=torch.float32)
            self.alpha_neg = torch.tensor(alpha[1], dtype=torch.float32)

    def forward(self, pred, true):
        # 1. Unified label format + smooth labeling
        if true.dim() == 1:
            true = F.one_hot(true.long(), num_classes=self.num_classes).float()
        true = true.to(pred.device)
        if self.label_smoothing > 0:
            true = (1 - self.label_smoothing) * true + self.label_smoothing / self.num_classes

        # 2. Device alignment
        self.alpha_pos = self.alpha_pos.to(pred.device)
        self.alpha_neg = self.alpha_neg.to(pred.device)
        self.gamma = self.gamma.to(pred.device)

        # 3. Basic BCE Loss
        loss = F.binary_cross_entropy_with_logits(pred / self.temperature, true, reduction="none")

        # 4. Calculate p_t
        pred_prob = torch.sigmoid(pred / self.temperature)
        pos_p_t = (true * pred_prob).sum(dim=1) / true.sum(dim=1).clamp(min=1e-6)
        neg_p_t = ((1 - true) * (1 - pred_prob)).sum(dim=1) / ((1 - true).sum(dim=1)).clamp(min=1e-6)

        # 5. Adaptive gamma (optimizing initial stability)
        mean_pos_pt = pos_p_t.mean()
        self.p_t_history.append(mean_pos_pt.item())  # deque automatically maintains its length, no need to manually pop

        # Fix 3: Initial warm-up mechanism (use current values when historical data is insufficient to avoid instability)
        warmup_steps = max(10, self.window_size // 2)  # Warm-up steps: at least 10 steps or half of the window
        if len(self.p_t_history) < warmup_steps:
            # Warm-up phase: Directly use the mean of p_t from the current batch to avoid fluctuations caused by a small number of historical values.
            p_t_smoothed = mean_pos_pt
        else:
            # Stabilization phase: SMA smoothing (historical 0.3 + current 0.7)
            p_t_old = sum(self.p_t_history) / len(self.p_t_history)
            p_t_smoothed = 0.3 * p_t_old + 0.7 * mean_pos_pt                          # Formula 2

        # Calculate gamma and apply the explicitly defined upper and lower limits
        gamma = -torch.log(p_t_smoothed.clamp(min=1e-6, max=0.999))
        gamma = gamma.clamp(min=self.gamma_clamp[0], max=self.gamma_clamp[1])          # Formula 3

        # 6. modulation factor
        pos_modulating = (1.0 - pos_p_t).pow(gamma).unsqueeze(1)
        neg_modulating = neg_p_t.pow(gamma).unsqueeze(1)
        modulating_factor = true * pos_modulating + (1 - true) * neg_modulating

        # 7. Class Balance Factor
        alpha_factor = true * self.alpha_pos.unsqueeze(0) + (1 - true) * self.alpha_neg.unsqueeze(0)

        # 8. Final loss
        loss = (loss * alpha_factor * modulating_factor).sum(dim=1).mean()             # Formula 4

        return loss


class v8ClassificationLoss:
    """Classification loss wrapper for multi-scale models (standardized parameter passing)"""

    def __init__(self, model_scale="N"):    #Select N/S/M/L/X here
        self.dtfl = None
        self.model_scale = model_scale
        print(f"v8ClassificationLoss initialized (scale: {model_scale})")

    def __call__(self, preds, batch):
        preds = preds[1] if isinstance(preds, (list, tuple)) else preds
        true = batch["cls"]

        if self.dtfl is None:
            num_classes = preds.shape[1]
            # Define parameters by scale (gamma_clamp is passed as a parameter instead of being added dynamically)
            if self.model_scale == "N":  # Keep the N model parameters unchanged
                self.dtfl = DynamicThresholdFocalLoss(
                    num_classes=num_classes,
                    gamma=1.5,
                    window_size=50,
                    temperature=0.8,
                    label_smoothing=0.1,
                    alpha=([0.7]*num_classes, [0.3]*num_classes),
                    gamma_clamp=(0.5, 2.0)  # Explicitly pass the upper and lower limits of gamma
                )
            elif self.model_scale == "S":  # S model optimization convergence
                self.dtfl = DynamicThresholdFocalLoss(
                    num_classes=num_classes,
                    gamma=1.4,
                    window_size=60,
                    temperature=0.85,
                    label_smoothing=0.08,
                    alpha=([0.68]*num_classes, [0.32]*num_classes),
                    gamma_clamp=(0.4, 1.8)  # Pass explicitly
                )
            elif self.model_scale == "M":
                self.dtfl = DynamicThresholdFocalLoss(
                    num_classes=num_classes,
                    gamma=1.3,
                    window_size=70,
                    temperature=0.9,
                    label_smoothing=0.06,
                    alpha=([0.65]*num_classes, [0.35]*num_classes),
                    gamma_clamp=(0.3, 1.6)
                )
            elif self.model_scale == "L":
                self.dtfl = DynamicThresholdFocalLoss(
                    num_classes=num_classes,
                    gamma=1.2,
                    window_size=80,
                    temperature=0.92,
                    label_smoothing=0.04,
                    alpha=([0.6]*num_classes, [0.4]*num_classes),
                    gamma_clamp=(0.2, 1.4)
                )
            elif self.model_scale == "X":
                self.dtfl = DynamicThresholdFocalLoss(
                    num_classes=num_classes,
                    gamma=1.1,
                    window_size=100,
                    temperature=0.95,
                    label_smoothing=0.02,
                    alpha=([0.55]*num_classes, [0.45]*num_classes),
                    gamma_clamp=(0.1, 1.2)
                )
            print(f"DTFL initialized for {self.model_scale} (num_classes={num_classes})")

        loss = self.dtfl(preds, true)
        loss_items = loss.detach()
        return loss, loss_items
