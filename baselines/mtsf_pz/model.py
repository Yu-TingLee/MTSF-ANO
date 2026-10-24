from torch import nn

from models.mtsf_ano import DataReuploadingVQC
from models.revin import RevIN


class Model(nn.Module):
    """MTSF-ANO with a fixed Pauli-Z readout on each qubit instead of ANO."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.C = configs.enc_in
        self.num_qubits = configs.num_qubits
        assert self.num_qubits == self.C, "MTSF-PZ requires num_qubits == enc_in"
        self.num_layers = configs.num_layers

        self.use_revin = bool(configs.use_revin)
        self.revin = RevIN(self.C) if self.use_revin else None
        self.input_proj = nn.Linear(self.C * self.seq_len, self.C)
        self.vqc = DataReuploadingVQC(
            self.C,
            self.num_layers,
            configs.QML_device,
            learnable_scaling=bool(configs.learnable_scaling),
            use_pauli_obs=True,
        )
        self.head = nn.Linear(self.vqc.num_obs, self.C * self.pred_len)

    def forward(self, x):
        # x: (B, L, C)
        B = x.shape[0]
        x = x.transpose(1, 2)
        x_norm = self.revin.normalize(x) if self.use_revin else x
        flat = x_norm.reshape(B, self.C * self.seq_len)
        h = self.input_proj(flat)  # (B, C)
        q_out = self.vqc(h).to(x.dtype)  # (B, C)
        y = self.head(q_out)
        y = y.reshape(B, self.C, self.pred_len)
        if self.use_revin:
            y = self.revin.denormalize(y)
        return y.transpose(1, 2)  # (B, H, C)
