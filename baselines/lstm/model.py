from torch import nn

from models.revin import RevIN


class Model(nn.Module):
    """LSTM over the window; the last hidden state is projected to all H steps."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.C = configs.enc_in
        self.hidden = configs.lstm_hidden
        self.num_layers = configs.lstm_layers
        dropout = configs.lstm_dropout

        self.lstm = nn.LSTM(
            input_size=self.C,
            hidden_size=self.hidden,
            num_layers=self.num_layers,
            batch_first=True,
            dropout=dropout if self.num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(self.hidden, self.C * self.pred_len)
        self.use_revin = bool(configs.use_revin)
        self.revin = RevIN(self.C) if self.use_revin else None

    def forward(self, x):
        # x: (B, L, C)
        B = x.shape[0]
        if self.use_revin:
            x = self.revin.normalize(x.transpose(1, 2)).transpose(1, 2)
        out, _ = self.lstm(x)  # (B, L, hidden)
        last = out[:, -1, :]  # (B, hidden)
        y = self.proj(last)  # (B, C * H)
        y = y.reshape(B, self.C, self.pred_len)
        if self.use_revin:
            y = self.revin.denormalize(y)
        return y.transpose(1, 2)  # (B, H, C)
