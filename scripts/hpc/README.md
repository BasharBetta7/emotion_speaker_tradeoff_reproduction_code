# HSE HPC Slurm Training

This directory contains a Slurm submit script for training StarGANv2-VC on the HSE cHARISMa supercomputer.

## Before Submitting

This script assumes the remote cluster already has your conda environment named `pt241`.

Make sure these paths from `Configs/config_emo.yml` exist on the cluster:

```text
Data/train_list_emo.txt
Data/val_list_emo.txt
Utils/JDC/bst.t7
Utils/ASR/config.yml
Utils/ASR/epoch_00100.pth
Models/exp2_emo_train_vanilla/epoch_00030.pth
```

The last checkpoint is required because `pretrained_model` is set in `Configs/config_emo.yml`.

## Submit training

From the repository root on the cluster:

```bash
sbatch --account=<project_id> scripts/hpc/train_emo.sbatch
```

If your project has only one active account, HSE Slurm may not require `--account`. If you participate in several projects, HSE says to specify it with `--account` or `-A`.

Override the config without editing the script:

```bash
sbatch --account=<project_id> \
  --export=ALL,CONFIG_PATH=Configs/config_emo_native_intensity.yml \
  scripts/hpc/train_emo.sbatch
```

Use the `test` partition for a short smoke run, then return to `normal` for real training:

```bash
sbatch --partition=test --time=00:20:00 --account=<project_id> scripts/hpc/train_emo.sbatch
```

Training logs go to:

```text
slurm-starganv2vc_emo-<job_id>.out
slurm-starganv2vc_emo-<job_id>.err
Models/exp2_emo_train_vanilla/train.log
Models/exp2_emo_train_vanilla/tensorboard/
```

## Monitor

```bash
squeue -u "$USER"
sacct -j <job_id>
```

You can also use HPC TaskMaster in the HSE supercomputer user cabinet for efficiency reports.
