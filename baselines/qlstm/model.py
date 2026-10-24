import pennylane as qml
import torch
from torch import nn

from models.revin import RevIN


# Adapted from Quantum_Long_Short_Term_Memory QLSTM_v0.py (see NOTICE).
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


def _q_function(x, q_weights, n_class):
    n_dep = q_weights.shape[-2]
    n_qub = q_weights.shape[-1]
    _H_layer(n_qub)
    _RY_layer(x)
    for k in range(n_dep):
        _entangling_layer(n_qub)
        _RY_layer(q_weights[..., k, :])
    return [qml.expval(qml.PauliZ(position)) for position in range(n_class)]


class GatedVQC(nn.Module):
    """The gate VQCs of a QLSTM cell, evaluated together with parameter broadcasting.

    Equivalent to n_gates separate VQCs with their own weights on the same input.
    """

    def __init__(self, vqc_depth, n_qubits, n_class, n_gates):
        super().__init__()
        self.weights = nn.Parameter(0.01 * torch.randn(n_gates, vqc_depth, n_qubits))
        self.n_gates = n_gates
        self.n_class = n_class
        self.dev = qml.device("default.qubit", wires=n_qubits)
        self.VQC = qml.QNode(
            _q_function, self.dev, interface="torch", diff_method="backprop"
        )

    def forward(self, X):
        # X: (B, n_qubits). Gates and batch share one broadcast axis of size G * B.
        B, nq = X.shape
        G, depth = self.n_gates, self.weights.shape[1]
        x_t = X.unsqueeze(0).expand(G, B, nq).reshape(G * B, nq)
        w_t = (
            self.weights.unsqueeze(1).expand(G, B, depth, nq).reshape(G * B, depth, nq)
        )
        out = torch.stack(self.VQC(x_t, w_t, self.n_class), dim=-1)
        return out.to(X.dtype).view(G, B, self.n_class)


class CustomQLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size, vqc_depth):
        super().__init__()
        self.hidden_size = hidden_size
        n_qubits = input_size + hidden_size
        # Input, forget, cell and output gates.
        self.gates = GatedVQC(
            vqc_depth=vqc_depth, n_qubits=n_qubits, n_class=hidden_size, n_gates=4
        )

    def forward(self, x, hidden):
        h_prev, c_prev = hidden
        combined = torch.cat((x, h_prev), dim=1)
        i_g, f_g, g_g, o_g = self.gates(combined)
        i_t = torch.sigmoid(i_g)
        f_t = torch.sigmoid(f_g)
        g_t = torch.tanh(g_g)
        o_t = torch.sigmoid(o_g)
        c_t = f_t * c_prev + i_t * g_t
        h_t = o_t * torch.tanh(c_t)
        return h_t, h_t, c_t


class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, lstm_cell_QT):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = lstm_cell_QT

    def forward(self, x, hidden=None):
        batch_size, seq_len, _ = x.size()
        if hidden is None:
            h_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
            c_t = torch.zeros(batch_size, self.hidden_size).to(x.device)
        else:
            h_t, c_t = hidden
        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :]
            out, h_t, c_t = self.cell(x_t, (h_t, c_t))
            outputs.append(out.unsqueeze(1))
        outputs = torch.cat(outputs, dim=1)
        return outputs, (h_t, c_t)


class Model(nn.Module):
    """Linear(C, qlstm_input) -> quantum LSTM -> linear head to all H steps."""

    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.C = configs.enc_in
        self.qlstm_input = configs.qlstm_input
        self.qlstm_hidden = configs.qlstm_hidden
        vqc_depth = configs.num_layers

        self.front = nn.Linear(self.C, self.qlstm_input)
        cell = CustomQLSTMCell(self.qlstm_input, self.qlstm_hidden, vqc_depth)
        self.qlstm = CustomLSTM(self.qlstm_input, self.qlstm_hidden, cell)
        self.proj = nn.Linear(self.qlstm_hidden, self.C * self.pred_len)
        self.use_revin = bool(configs.use_revin)
        self.revin = RevIN(self.C) if self.use_revin else None

    def forward(self, x):
        # x: (B, L, C)
        B = x.shape[0]
        if self.use_revin:
            x = self.revin.normalize(x.transpose(1, 2)).transpose(1, 2)
        seq = self.front(x)  # (B, L, qlstm_input)
        out, _ = self.qlstm(seq)  # (B, L, qlstm_hidden)
        last = out[:, -1, :]  # (B, qlstm_hidden)
        y = self.proj(last).reshape(B, self.C, self.pred_len)
        if self.use_revin:
            y = self.revin.denormalize(y)
        return y.transpose(1, 2)  # (B, H, C)
