import argparse
import random

import numpy as np
import torch

from exp.experiment import Experiment


def parse_args():
    parser = argparse.ArgumentParser(description="MTSF-ANO")

    parser.add_argument("--random_seed", type=int, default=2021)
    parser.add_argument(
        "--is_training",
        type=int,
        default=1,
        help="0 evaluates a saved checkpoint instead of training",
    )
    parser.add_argument("--model", type=str, default="MTSF-ANO")

    # VQC models
    parser.add_argument("--num_qubits", type=int, default=7)
    parser.add_argument("--QML_device", type=str, default="default.qubit")
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--use_revin", type=int, default=1)
    parser.add_argument("--learnable_scaling", type=int, default=1)
    parser.add_argument("--use_pauli_obs", type=int, default=0)
    parser.add_argument("--n_local", type=int, default=1, help="ANO locality k")
    parser.add_argument(
        "--entanglement", type=str, default="cnot", choices=["cnot", "none"]
    )

    # baselines
    parser.add_argument(
        "--individual", type=int, default=0, help="DLinear: one layer per channel"
    )
    parser.add_argument("--lstm_hidden", type=int, default=64)
    parser.add_argument("--lstm_layers", type=int, default=2)
    parser.add_argument("--lstm_dropout", type=float, default=0.05)
    parser.add_argument("--qlstm_input", type=int, default=3)
    parser.add_argument("--qlstm_hidden", type=int, default=4)
    parser.add_argument("--qfwp_qubits", type=int, default=7)
    parser.add_argument("--qfwp_depth", type=int, default=3)
    parser.add_argument("--qfwp_a", type=int, default=7)

    # data
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--root_path", type=str, default="./dataset/ETT-small/")
    parser.add_argument("--data_path", type=str, default="ETTh1.csv")
    parser.add_argument("--features", type=str, default="M")
    parser.add_argument("--target", type=str, default="OT")
    parser.add_argument("--freq", type=str, default="h")
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/")
    parser.add_argument("--no_checkpoint", type=int, default=0)
    parser.add_argument("--save_suffix", type=str, default="")

    # forecasting
    parser.add_argument("--seq_len", type=int, default=16, help="lookback length L")
    parser.add_argument("--label_len", type=int, default=0)
    parser.add_argument("--pred_len", type=int, default=1, help="horizon H")
    parser.add_argument("--enc_in", type=int, default=7, help="number of channels")

    # optimization
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--lradj", type=str, default="constant")
    parser.add_argument("--pct_start", type=float, default=0.3)

    # gpu
    parser.add_argument("--use_gpu", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--use_multi_gpu", action="store_true", default=False)
    parser.add_argument("--devices", type=str, default="0,1")

    return parser.parse_args()


def run_name(args):
    return f"{args.data}_L{args.seq_len}_H{args.pred_len}_s{args.random_seed}"


def main():
    args = parse_args()

    if not args.use_revin and not args.save_suffix:
        args.save_suffix = "-noRevIN"

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    args.use_gpu = bool(torch.cuda.is_available() and args.use_gpu)
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(" ", "")
        args.device_ids = [int(x) for x in args.devices.split(",")]
        args.gpu = args.device_ids[0]

    print("Args:", args)

    setting = run_name(args)
    exp = Experiment(args)
    if args.is_training:
        print(f">>> training: {setting}")
        exp.train(setting)
        print(f">>> testing: {setting}")
        exp.test(setting)
    else:
        print(f">>> testing only: {setting}")
        exp.test(setting, load_checkpoint=True)
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
