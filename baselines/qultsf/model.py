import pennylane as qml
from torch import nn

from models.revin import RevIN


# From QuLTSF models/QuLTSF.py (see NOTICE).
class Hybrid_QML_Model(nn.Module):
    def __init__(
        self,
        lookback_window_size,
        forecast_window_size,
        num_qubits,
        QML_device,
        num_layers,
    ):
        super().__init__()
        self.forecast_window_size = forecast_window_size

        self.dev = qml.device(QML_device, wires=num_qubits)

        @qml.qnode(self.dev, interface="torch", diff_method="best")
        def quantum_function(inputs, weights):
            qml.AmplitudeEmbedding(
                features=inputs, wires=range(num_qubits), normalize=True
            )
            qml.StronglyEntanglingLayers(weights=weights, wires=range(num_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

        self._quantum_circuit = quantum_function
        q_weights_shape = {"weights": (num_layers, num_qubits, 3)}

        self.input_classical_layer = nn.Linear(lookback_window_size, 2**num_qubits)
        self.hidden_quantum_layer = qml.qnn.TorchLayer(
            self._quantum_circuit, q_weights_shape
        )
        self.output_classical_layer = nn.Linear(num_qubits, forecast_window_size)

    def forward(self, batch_input):
        # batch_input: (B, C, L)
        y = batch_input.reshape(
            batch_input.shape[0] * batch_input.shape[1], batch_input.shape[2]
        )
        y = self.input_classical_layer(y)
        y = self.hidden_quantum_layer(y)
        y = self.output_classical_layer(y)
        return y.reshape(
            batch_input.shape[0], batch_input.shape[1], self.forecast_window_size
        )  # (B, C, H)


class _QuLTSF(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.num_qubits = configs.num_qubits
        self.QML_device = configs.QML_device
        self.num_layers = configs.num_layers
        self.hybrid_qml_model = Hybrid_QML_Model(
            self.seq_len,
            self.pred_len,
            self.num_qubits,
            self.QML_device,
            self.num_layers,
        )

    def forward(self, x):
        # x: (B, L, C)
        return self.hybrid_qml_model(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, H, C)


class Model(_QuLTSF):
    """QuLTSF with RevIN around it."""

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
