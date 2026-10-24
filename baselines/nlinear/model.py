from torch import nn

from models.revin import RevIN


# From LTSF-Linear models/NLinear.py (see NOTICE).
class _NLinear(nn.Module):
    """Normalization-Linear."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x):
        # x: (B, L, C)
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = x + seq_last
        return x  # (B, H, C)


class Model(_NLinear):
    """NLinear with RevIN around it."""

    def __init__(self, configs):
        super().__init__(configs)
        self.use_revin = bool(configs.use_revin)
        self.revin = RevIN(configs.enc_in) if self.use_revin else None

    def forward(self, x):
        # x: (B, L, C)
        if not self.use_revin:
            return super().forward(x)
        x = self.revin.normalize(x.transpose(1, 2)).transpose(1, 2)
        y = super().forward(x)
        y = self.revin.denormalize(y.transpose(1, 2))
        return y.transpose(1, 2)  # (B, H, C)
