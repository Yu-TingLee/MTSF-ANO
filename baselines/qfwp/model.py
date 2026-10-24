import pennylane as qml
import torch
from torch import nn

from models.revin import RevIN


# Adapted from Quantum_FWP vqFWP_time_series.py (see NOTICE).
def _H_layer(nqubits):
    for idx in range(nqubits):
        qml.Hadamard(wires=idx)


def _RY_layer(w):
    for idx in range(w.shape[-1]):
        qml.RY(w[..., idx], wires=idx)


def _entangling_layer(nqubits):
    for i in range(0, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])
    for i in range(1, nqubits - 1, 2):
        qml.CNOT(wires=[i, i + 1])


def _quantum_net(inputs, q_weights, n_outputs):
    # q_weights: (..., n_dep, n_qub), inputs: (..., n_inp)
    n_dep = q_weights.shape[-2]
    n_qub = q_weights.shape[-1]
    _H_layer(n_qub)
    _RY_layer(inputs)
    for k in range(n_dep):
        _entangling_layer(n_qub)
        _RY_layer(q_weights[..., k, :])
    return [qml.expval(qml.PauliZ(position)) for position in range(n_outputs)]


class FWPCell(nn.Module):
    """Slow programmer + VQC. Emits an a_dim measurement and the updated program."""

    def __init__(self, s_dim, a_dim, n_qubits, q_depth):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_depth = q_depth
        self.a_dim = a_dim

        self.slow_program_encoder = nn.Linear(s_dim, n_qubits)
        self.slow_program_layer_idx = nn.Linear(n_qubits, q_depth)
        self.slow_program_qubit_idx = nn.Linear(n_qubits, n_qubits)

        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.VQC = qml.QNode(
            _quantum_net, self.dev, interface="torch", diff_method="backprop"
        )

    def forward(self, batch_item, previous_circuit_param):
        # batch_item: (B, s_dim); previous_circuit_param: (B, q_depth, n_qubits)
        res = self.slow_program_encoder(batch_item)  # (B, n_qubits)
        res_layer_idx = self.slow_program_layer_idx(res)  # (B, q_depth)
        res_qubit_idx = self.slow_program_qubit_idx(res)  # (B, n_qubits)

        # Fast-weight update: per-sample outer product added to the running program.
        out_circuit_params = torch.einsum("bi,bj->bij", res_layer_idx, res_qubit_idx)
        out_circuit_params = out_circuit_params + previous_circuit_param

        out = torch.stack(self.VQC(batch_item, out_circuit_params, self.a_dim), dim=-1)
        return out.to(
            batch_item.dtype
        ), out_circuit_params  # (B, a_dim), (B, q_depth, n_qubits)

    def initial_fast_params(self, batch_size, device):
        return torch.zeros(batch_size, self.q_depth, self.n_qubits, device=device)


class FWP(nn.Module):
    """Rolls the FWPCell over the sequence. Input: (B, L, s_dim)."""

    def __init__(self, s_dim, a_dim, n_qubits, q_depth):
        super().__init__()
        self.cell = FWPCell(s_dim, a_dim, n_qubits, q_depth)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        fast_params = self.cell.initial_fast_params(batch_size, x.device)
        outputs = []
        for t in range(seq_len):
            out, fast_params = self.cell(x[:, t, :], fast_params)
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)  # (B, L, a_dim)
        return outputs


class Model(nn.Module):
    """Quantum FWP over the window; the last output is projected to all H steps."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.C = configs.enc_in
        n_qubits = configs.qfwp_qubits
        q_depth = configs.qfwp_depth
        a_dim = configs.qfwp_a
        assert self.C <= n_qubits, (
            "qfwp_qubits must be >= enc_in (channels are RY-embedded)"
        )

        self.fwp = FWP(s_dim=self.C, a_dim=a_dim, n_qubits=n_qubits, q_depth=q_depth)
        self.proj = nn.Linear(a_dim, self.C * self.pred_len)
        self.use_revin = bool(configs.use_revin)
        self.revin = RevIN(self.C) if self.use_revin else None

    def forward(self, x):
        # x: (B, L, C)
        B = x.shape[0]
        if self.use_revin:
            x = self.revin.normalize(x.transpose(1, 2)).transpose(1, 2)
        out = self.fwp(x)  # (B, L, a_dim)
        last = out[:, -1, :]  # (B, a_dim)
        y = self.proj(last).reshape(B, self.C, self.pred_len)
        if self.use_revin:
            y = self.revin.denormalize(y)
        return y.transpose(1, 2)  # (B, H, C)
