from torch import nn

from models.mtsf_ano import DataReuploadingVQC
from models.revin import RevIN


class Model(nn.Module):
    """Channel-independent MTSF-ANO: every channel runs through one shared VQC."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.C = configs.enc_in
        self.num_qubits = configs.num_qubits
        assert self.num_qubits == self.C, "ANO requires num_qubits == enc_in"
        self.num_layers = configs.num_layers

        self.use_revin = bool(configs.use_revin)
        self.revin = RevIN(self.C) if self.use_revin else None
        self.input_proj = nn.Linear(self.seq_len, self.num_qubits)
        self.vqc = DataReuploadingVQC(
            self.num_qubits,
            self.num_layers,
            configs.QML_device,
            learnable_scaling=bool(configs.learnable_scaling),
            use_pauli_obs=bool(configs.use_pauli_obs),
            n_local=configs.n_local,
            entanglement=configs.entanglement,
        )
        self.head = nn.Linear(self.vqc.num_obs, self.pred_len)

    def forward(self, x):
        # x: (B, L, C)
        B, _, C = x.shape
        x = x.transpose(1, 2)
        x_norm = self.revin.normalize(x) if self.use_revin else x
        y = x_norm.reshape(B * C, self.seq_len)
        h = self.input_proj(y)  # (B*C, num_qubits)
        q_out = self.vqc(h).to(x.dtype)  # (B*C, M)
        y = self.head(q_out)  # (B*C, H)
        y = y.reshape(B, C, self.pred_len)
        if self.use_revin:
            y = self.revin.denormalize(y)
        return y.transpose(1, 2)  # (B, H, C)
