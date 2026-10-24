import itertools

import pennylane as qml
import torch
from torch import nn

from models.revin import RevIN


class ANOHermitians(nn.Module):
    """M trainable K x K Hermitian observables, K = 2^n_local.

    Only the free parameters are stored: K real diagonal entries and the
    K(K-1)/2 complex entries above the diagonal.
    """

    def __init__(self, M, K, init_scale=0.1):
        super().__init__()
        self.M = M
        self.K = K
        iu = torch.triu_indices(K, K, offset=1)
        self.register_buffer("iu_row", iu[0])
        self.register_buffer("iu_col", iu[1])
        num_upper = iu.shape[1]
        self.diag = nn.Parameter(torch.randn(M, K) * init_scale)
        self.upper_real = nn.Parameter(torch.randn(M, num_upper) * init_scale)
        self.upper_imag = nn.Parameter(torch.randn(M, num_upper) * init_scale)

    def assemble(self):
        real = torch.zeros(
            self.M, self.K, self.K, device=self.diag.device, dtype=self.diag.dtype
        )
        imag = torch.zeros_like(real)
        real[:, self.iu_row, self.iu_col] = self.upper_real
        imag[:, self.iu_row, self.iu_col] = self.upper_imag
        real_part = real + real.transpose(-1, -2) + torch.diag_embed(self.diag)
        imag_part = imag - imag.transpose(-1, -2)
        return torch.complex(real_part, imag_part)  # (M, K, K)


class DataReuploadingVQC(nn.Module):
    """Hardware-efficient data re-uploading circuit.

    Each layer applies RZ-RY variational rotations, an optional CNOT ring and
    RY-RZ re-uploading of the inputs. The readout is either <Z> per qubit or
    one trainable Hermitian per n_local-subset of qubits (ANO).
    """

    def __init__(
        self,
        num_qubits,
        num_layers,
        qml_device="default.qubit",
        learnable_scaling=True,
        use_pauli_obs=False,
        n_local=1,
        entanglement="cnot",
    ):
        super().__init__()
        self.C = num_qubits
        self.D = num_layers
        self.K = 2**num_qubits
        self.learnable_scaling = learnable_scaling
        self.use_pauli_obs = use_pauli_obs
        assert entanglement in ("cnot", "none"), f"unknown entanglement: {entanglement}"
        self.entanglement = entanglement

        self.n_var = 2
        self.n_enc = 2
        self.var_weights = nn.Parameter(torch.randn(self.D, self.C, self.n_var) * 0.1)
        if learnable_scaling:
            self.scale_weights = nn.Parameter(torch.ones(self.D, self.C, self.n_enc))
        else:
            self.register_buffer(
                "scale_weights", torch.ones(self.D, self.C, self.n_enc)
            )

        if use_pauli_obs:
            self.num_obs = self.C
        else:
            assert 1 <= n_local <= self.C, f"n_local must be in [1, {self.C}]"
            self.n_local = n_local
            self.K_obs = 2**n_local
            self.subsets = list(itertools.combinations(range(self.C), n_local))
            self.num_obs = len(self.subsets)
            self.observables = ANOHermitians(self.num_obs, self.K_obs)

        dev = qml.device(qml_device, wires=self.C)
        C = self.C
        D = self.D
        use_entanglement = self.entanglement != "none"

        def circuit_body(inputs, var_w, scale_w):
            for q in range(C):
                qml.Hadamard(wires=q)
            for d in range(D):
                for q in range(C):
                    qml.RZ(var_w[d, q, 0], wires=q)
                    qml.RY(var_w[d, q, 1], wires=q)
                if use_entanglement:
                    for q in range(C):
                        qml.CNOT(wires=[q, (q + 1) % C])
                for q in range(C):
                    qml.RY(scale_w[d, q, 0] * inputs[..., q], wires=q)
                    qml.RZ(scale_w[d, q, 1] * inputs[..., q], wires=q)

        if use_pauli_obs:

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def qexpval_z(inputs, var_w, scale_w):
                circuit_body(inputs, var_w, scale_w)
                return [qml.expval(qml.PauliZ(q)) for q in range(C)]

            self.qexpval_z = qexpval_z
        else:
            subsets = self.subsets

            @qml.qnode(dev, interface="torch", diff_method="backprop")
            def qexpval(inputs, var_w, scale_w, herms):
                circuit_body(inputs, var_w, scale_w)
                return [
                    qml.expval(qml.Hermitian(herms[q], wires=list(S)))
                    for q, S in enumerate(subsets)
                ]

            self.qexpval = qexpval

    def forward(self, h):
        # h: (B, C) -> (B, num_obs)
        if self.use_pauli_obs:
            exps = self.qexpval_z(h, self.var_weights, self.scale_weights)
            return torch.stack(exps, dim=-1)

        herms = self.observables.assemble()
        exps = self.qexpval(h, self.var_weights, self.scale_weights, herms)
        return torch.stack(exps, dim=-1)


class Model(nn.Module):
    """MTSF-ANO: RevIN -> linear projection -> re-uploading VQC with ANO -> linear head."""

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
        self.input_proj = nn.Linear(self.C * self.seq_len, self.C)
        self.vqc = DataReuploadingVQC(
            self.C,
            self.num_layers,
            configs.QML_device,
            learnable_scaling=bool(configs.learnable_scaling),
            use_pauli_obs=bool(configs.use_pauli_obs),
            n_local=configs.n_local,
            entanglement=configs.entanglement,
        )
        self.head = nn.Linear(self.vqc.num_obs, self.C * self.pred_len)

    def forward(self, x):
        # x: (B, L, C)
        B = x.shape[0]
        x = x.transpose(1, 2)
        x_norm = self.revin.normalize(x) if self.use_revin else x
        flat = x_norm.reshape(B, self.C * self.seq_len)
        h = self.input_proj(flat)  # (B, C)
        q_out = self.vqc(h).to(x.dtype)  # (B, M)
        y = self.head(q_out)
        y = y.reshape(B, self.C, self.pred_len)
        if self.use_revin:
            y = self.revin.denormalize(y)
        return y.transpose(1, 2)  # (B, H, C)
