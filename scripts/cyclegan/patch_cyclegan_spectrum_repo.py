#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


PATCH_MARKER = "# patched by StarGANv2-VC CycleGAN spectrum frontier pipeline"


def replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"Expected exactly one {label} occurrence, found {count}.")
    return text.replace(old, new, 1)


def patch_train_py(train_path: Path) -> bool:
    text = train_path.read_text(encoding="utf-8")
    if PATCH_MARKER in text:
        return False

    text = replace_once(
        text,
        "def train(train_A_dir, train_B_dir, model_dir, model_name, random_seed, validation_A_dir, validation_B_dir, output_dir, tensorboard_log_dir):",
        "def train(train_A_dir, train_B_dir, model_dir, model_name, random_seed, validation_A_dir, validation_B_dir, output_dir, tensorboard_log_dir, num_epochs_override=None, max_samples_override=None):",
        "train signature",
    )
    text = replace_once(
        text,
        "    num_epochs = 5000",
        "    num_epochs = int(num_epochs_override) if num_epochs_override is not None else 5000",
        "num_epochs assignment",
    )
    text = replace_once(
        text,
        "    max_samples = 1000",
        "    max_samples = int(max_samples_override) if max_samples_override is not None else 1000",
        "max_samples assignment",
    )
    text = replace_once(
        text,
        "    parser.add_argument('--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.', default = tensorboard_log_dir_default)",
        (
            "    parser.add_argument('--tensorboard_log_dir', type = str, help = 'TensorBoard log directory.', default = tensorboard_log_dir_default)\n"
            "    parser.add_argument('--num_epochs', type = int, default = None, help = 'Override spectrum CycleGAN training epochs.')\n"
            "    parser.add_argument('--max_samples', type = int, default = None, help = 'Override sampled training windows per epoch.')"
        ),
        "argparse insertion point",
    )
    text = replace_once(
        text,
        "    tensorboard_log_dir = argv.tensorboard_log_dir",
        (
            "    tensorboard_log_dir = argv.tensorboard_log_dir\n"
            "    num_epochs_override = argv.num_epochs\n"
            "    max_samples_override = argv.max_samples"
        ),
        "argv extraction point",
    )
    text = replace_once(
        text,
        "    train(train_A_dir = train_A_dir, train_B_dir = train_B_dir, model_dir = model_dir, model_name = model_name, random_seed = random_seed, validation_A_dir = validation_A_dir, validation_B_dir = validation_B_dir, output_dir = output_dir, tensorboard_log_dir = tensorboard_log_dir)",
        (
            "    train(train_A_dir = train_A_dir, train_B_dir = train_B_dir, model_dir = model_dir, model_name = model_name, random_seed = random_seed, validation_A_dir = validation_A_dir, validation_B_dir = validation_B_dir, output_dir = output_dir, tensorboard_log_dir = tensorboard_log_dir, num_epochs_override = num_epochs_override, max_samples_override = max_samples_override)"
        ),
        "train call",
    )
    text = f"{PATCH_MARKER}\n{text}"
    train_path.write_text(text, encoding="utf-8")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch the external KunZhou CycleGAN-CWT repo for configurable spectrum-only training."
    )
    parser.add_argument("--repo-path", required=True, help="Path to cloned external CycleGAN-CWT repository.")
    args = parser.parse_args()

    repo_path = Path(args.repo_path).resolve()
    train_path = repo_path / "train.py"
    if not train_path.exists():
        raise FileNotFoundError(f"Missing train.py under external repo: {train_path}")

    changed = patch_train_py(train_path)
    if changed:
        print(f"Patched configurable spectrum training: {train_path}")
    else:
        print(f"Already patched: {train_path}")


if __name__ == "__main__":
    main()
