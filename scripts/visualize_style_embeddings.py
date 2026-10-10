#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path
import sys
import tempfile

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "starganv2vc-mpl"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "starganv2vc-xdg"))

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd
import soundfile as sf
try:
    import seaborn as sns
except ModuleNotFoundError:
    sns = None
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, silhouette_samples, silhouette_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
import torchaudio
from tqdm import tqdm

from eval_mel_viz import load_eval_bundle


EMOTION_TO_LABEL = {
    "Angry": 0,
    "Happy": 1,
    "Neutral": 2,
    "Sad": 3,
    "Surprise": 4,
}

EMOTION_CANONICAL = {key.lower(): key for key in EMOTION_TO_LABEL}
EMOTION_ORDER = list(EMOTION_TO_LABEL.keys())
EMOTION_PALETTE = {
    "Angry": "#d1495b",
    "Happy": "#edae49",
    "Neutral": "#7f8c8d",
    "Sad": "#00798c",
    "Surprise": "#6a4c93",
}

SPEAKER_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*", "h", "8"]
REQUIRED_COLUMNS = {"file_id", "speaker_id", "emotion", "wav_path"}
RANK_COLUMN_PREFIX = "rank_"


def resolve_path(path_str: str, repo_root: Path) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def canonicalize_emotion(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in EMOTION_CANONICAL:
        raise ValueError(f"Unsupported emotion label: {value}")
    return EMOTION_CANONICAL[normalized]


def load_input_frame(input_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(input_csv, sep=None, engine="python")
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {sorted(missing)}")

    out = df.copy()
    out["emotion"] = out["emotion"].map(canonicalize_emotion)
    out["label"] = out["emotion"].map(EMOTION_TO_LABEL)
    out["speaker_id"] = out["speaker_id"].astype(str)
    out["wav_path"] = out["wav_path"].astype(str)
    return out


def sample_frame(
    df: pd.DataFrame,
    max_samples: int | None,
    samples_per_emotion: int | None,
    samples_per_speaker: int | None,
    random_seed: int,
) -> pd.DataFrame:
    shuffled = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    kept_rows: list[dict] = []
    emotion_counts: Counter[str] = Counter()
    speaker_counts: Counter[str] = Counter()

    for row in shuffled.to_dict("records"):
        emotion = row["emotion"]
        speaker = row["speaker_id"]

        if samples_per_emotion is not None and emotion_counts[emotion] >= samples_per_emotion:
            continue
        if samples_per_speaker is not None and speaker_counts[speaker] >= samples_per_speaker:
            continue

        kept_rows.append(row)
        emotion_counts[emotion] += 1
        speaker_counts[speaker] += 1

        if max_samples is not None and len(kept_rows) >= max_samples:
            break

    if not kept_rows:
        raise ValueError("Sampling removed every row. Relax the sampling constraints.")

    sampled = pd.DataFrame(kept_rows)
    return sampled.sort_values(["emotion", "speaker_id", "file_id"]).reset_index(drop=True)


def load_audio_mono(wav_path: str, sample_rate: int) -> torch.Tensor:
    try:
        wav, sr = torchaudio.load(wav_path)
    except (ImportError, OSError, RuntimeError):
        wav_np, sr = sf.read(wav_path, dtype="float32")
        wav = torch.from_numpy(wav_np)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.transpose(0, 1)
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != sample_rate:
        wav = torchaudio.functional.resample(wav, sr, sample_rate)
    return wav


def waveform_to_mel(wav: torch.Tensor, bundle) -> torch.Tensor:
    wav = wav.to(bundle.device)
    mel = bundle.to_mel(wav).squeeze(0)
    mel = (torch.log(1e-5 + mel) - bundle.mean) / bundle.std
    return mel


def pad_mels(mels: list[torch.Tensor], device: str) -> torch.Tensor:
    mel_bins = mels[0].size(0)
    max_frames = max(mel.size(1) for mel in mels)
    batch = torch.zeros((len(mels), 1, mel_bins, max_frames), dtype=torch.float32, device=device)
    for idx, mel in enumerate(mels):
        frames = mel.size(1)
        batch[idx, 0, :, :frames] = mel
    return batch


@torch.no_grad()
def extract_embeddings(bundle, df: pd.DataFrame, batch_size: int) -> pd.DataFrame:
    records: list[dict] = []
    checkpoint_name = getattr(bundle, "checkpoint_name", "checkpoint")
    has_rank_head = hasattr(bundle.model, "rank_head")

    for start in tqdm(range(0, len(df), batch_size), desc="Extracting style embeddings"):
        batch_df = df.iloc[start:start + batch_size]
        mels: list[torch.Tensor] = []
        labels: list[int] = []

        for row in batch_df.itertuples(index=False):
            wav = load_audio_mono(row.wav_path, bundle.sr)
            mels.append(waveform_to_mel(wav, bundle))
            labels.append(int(row.label))

        x_ref = pad_mels(mels, bundle.device)
        y_ref = torch.tensor(labels, dtype=torch.long, device=bundle.device)
        style_embeddings = bundle.model.style_encoder(x_ref, y_ref)
        embeddings = style_embeddings.detach().cpu().numpy()
        rank_scores = None
        if has_rank_head:
            rank_scores = bundle.model.rank_head(style_embeddings).detach().cpu().numpy()

        for item_idx, (row, embedding) in enumerate(zip(batch_df.itertuples(index=False), embeddings)):
            record = {
                "file_id": row.file_id,
                "speaker_id": row.speaker_id,
                "emotion": row.emotion,
                "label": int(row.label),
                "wav_path": row.wav_path,
                "checkpoint": checkpoint_name,
            }
            for idx, value in enumerate(embedding):
                record[f"emb_{idx:03d}"] = float(value)
            if rank_scores is not None:
                for emotion, label in EMOTION_TO_LABEL.items():
                    record[f"{RANK_COLUMN_PREFIX}{emotion.lower()}"] = float(rank_scores[item_idx, label])
                record["rank_target"] = float(rank_scores[item_idx, int(row.label)])
            records.append(record)

    return pd.DataFrame(records)


def get_embedding_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith("emb_")]


def get_rank_columns(df: pd.DataFrame) -> list[str]:
    return [column for column in df.columns if column.startswith(RANK_COLUMN_PREFIX)]


def standardize_embeddings(raw_df: pd.DataFrame) -> tuple[pd.DataFrame, list[str], object]:
    embedding_columns = get_embedding_columns(raw_df)
    scaler = StandardScaler()
    standardized = scaler.fit_transform(raw_df[embedding_columns].to_numpy())
    return raw_df[["file_id", "speaker_id", "emotion", "label", "wav_path", "checkpoint"]].copy(), embedding_columns, standardized


def component_columns(dims: int) -> list[str]:
    columns = ["x", "y"]
    if dims == 3:
        columns.append("z")
    return columns


def compute_pca_projection(raw_df: pd.DataFrame, dims: int) -> tuple[pd.DataFrame, str]:
    projection_df, _, standardized = standardize_embeddings(raw_df)
    pca = PCA(n_components=dims)
    coords = pca.fit_transform(standardized)

    projection_df["method"] = "pca"
    projection_df["dims"] = dims
    for idx, column in enumerate(component_columns(dims)):
        projection_df[column] = coords[:, idx]
    variance = pca.explained_variance_ratio_
    title_suffix = " | ".join(
        f"PC{idx + 1} {variance[idx] * 100:.1f}%"
        for idx in range(dims)
    )
    return projection_df, title_suffix


def compute_umap_projection(raw_df: pd.DataFrame, random_seed: int, dims: int) -> tuple[pd.DataFrame, str] | None:
    try:
        import umap
    except Exception:
        return None

    projection_df, _, standardized = standardize_embeddings(raw_df)
    reducer = umap.UMAP(n_components=dims, random_state=random_seed)
    coords = reducer.fit_transform(standardized)

    projection_df["method"] = "umap"
    projection_df["dims"] = dims
    for idx, column in enumerate(component_columns(dims)):
        projection_df[column] = coords[:, idx]
    return projection_df, f"UMAP {dims}D projection"


def add_emotion_centroids(ax, projection_df: pd.DataFrame) -> None:
    centroids = (
        projection_df.groupby("emotion", sort=False)[["x", "y"]]
        .mean()
        .reset_index()
    )
    for row in centroids.itertuples(index=False):
        ax.scatter(
            row.x,
            row.y,
            s=220,
            marker="X",
            color=EMOTION_PALETTE[row.emotion],
            edgecolors="black",
            linewidths=0.9,
            zorder=5,
        )
        ax.text(row.x, row.y, f" {row.emotion}", fontsize=10, weight="bold", va="bottom", ha="left")


def add_emotion_centroids_3d(ax, projection_df: pd.DataFrame) -> None:
    centroids = (
        projection_df.groupby("emotion", sort=False)[["x", "y", "z"]]
        .mean()
        .reset_index()
    )
    for row in centroids.itertuples(index=False):
        ax.scatter(
            row.x,
            row.y,
            row.z,
            s=240,
            marker="X",
            color=EMOTION_PALETTE[row.emotion],
            edgecolors="black",
            linewidths=0.9,
            depthshade=False,
        )
        ax.text(row.x, row.y, row.z, f" {row.emotion}", fontsize=9)


def get_emotion_centroids_3d(projection_df: pd.DataFrame) -> pd.DataFrame:
    return (
        projection_df.groupby("emotion", sort=False)[["x", "y", "z"]]
        .mean()
        .reset_index()
    )


def save_emotion_plot_2d(projection_df: pd.DataFrame, method_title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    for emotion in EMOTION_ORDER:
        subset = projection_df[projection_df["emotion"] == emotion]
        if subset.empty:
            continue
        ax.scatter(
            subset["x"],
            subset["y"],
            s=40,
            alpha=0.72,
            linewidths=0,
            color=EMOTION_PALETTE[emotion],
            label=emotion,
        )
    add_emotion_centroids(ax, projection_df)
    checkpoint_name = projection_df["checkpoint"].iloc[0]
    ax.set_title(f"Style Encoder Embedding Space by Emotion\n{checkpoint_name} | {method_title}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.25)
    ax.legend(title="Emotion")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def build_speaker_marker_map(projection_df: pd.DataFrame) -> dict[str, str]:
    speakers = sorted(projection_df["speaker_id"].unique())
    return {
        speaker: SPEAKER_MARKERS[idx % len(SPEAKER_MARKERS)]
        for idx, speaker in enumerate(speakers)
    }


def build_speaker_color_map(projection_df: pd.DataFrame) -> dict[str, tuple[float, float, float]]:
    speakers = sorted(projection_df["speaker_id"].unique())
    if sns is not None:
        palette = sns.color_palette("tab20", n_colors=max(len(speakers), 1))
    else:
        cmap = plt.get_cmap("tab20")
        palette = [cmap(idx % cmap.N) for idx in range(max(len(speakers), 1))]
    return {
        speaker: palette[idx]
        for idx, speaker in enumerate(speakers)
    }


def draw_boxplot_by_emotion(
    ax: plt.Axes,
    data: pd.DataFrame,
    y_column: str,
    order: list[str],
) -> None:
    if sns is not None:
        sns.boxplot(
            data=data,
            x="emotion",
            y=y_column,
            order=order,
            palette=EMOTION_PALETTE,
            ax=ax,
        )
        return

    values = [
        data.loc[data["emotion"] == emotion, y_column].dropna().to_numpy()
        for emotion in order
    ]
    box = ax.boxplot(values, labels=order, patch_artist=True)
    for patch, emotion in zip(box["boxes"], order):
        patch.set_facecolor(EMOTION_PALETTE[emotion])


def save_speaker_plot_2d(projection_df: pd.DataFrame, method_title: str, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 8))
    speaker_color_map = build_speaker_color_map(projection_df)

    for speaker_id in sorted(projection_df["speaker_id"].unique()):
        speaker_df = projection_df[projection_df["speaker_id"] == speaker_id]
        for emotion in EMOTION_ORDER:
            subset = speaker_df[speaker_df["emotion"] == emotion]
            if subset.empty:
                continue
            ax.scatter(
                subset["x"],
                subset["y"],
                s=38,
                alpha=0.68,
                color=speaker_color_map[speaker_id],
                marker="o",
                linewidths=0,
            )

    add_emotion_centroids(ax, projection_df)
    checkpoint_name = projection_df["checkpoint"].iloc[0]
    ax.set_title(f"Style Encoder Embedding Space by Speaker\n{checkpoint_name} | {method_title}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(alpha=0.25)

    emotion_handles = [
        Line2D([0], [0], marker="o", color="w", label=emotion, markerfacecolor=EMOTION_PALETTE[emotion], markersize=8)
        for emotion in EMOTION_ORDER
    ]
    speaker_handles = [
        Line2D([0], [0], marker="o", color="w", linestyle="", label=str(speaker), markerfacecolor=color, markersize=7)
        for speaker, color in speaker_color_map.items()
    ]
    emotion_legend = ax.legend(handles=emotion_handles, title="Emotion", loc="upper left", frameon=True)
    ax.add_artist(emotion_legend)
    ax.legend(handles=speaker_handles, title="Speaker", loc="upper right", ncol=2, frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_emotion_plot_3d(projection_df: pd.DataFrame, method_title: str, output_path: Path) -> None:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    for emotion in EMOTION_ORDER:
        subset = projection_df[projection_df["emotion"] == emotion]
        if subset.empty:
            continue
        ax.scatter(
            subset["x"],
            subset["y"],
            subset["z"],
            s=28,
            alpha=0.7,
            color=EMOTION_PALETTE[emotion],
            label=emotion,
            depthshade=True,
        )

    add_emotion_centroids_3d(ax, projection_df)
    checkpoint_name = projection_df["checkpoint"].iloc[0]
    ax.set_title(f"Style Encoder Embedding Space by Emotion\n{checkpoint_name} | {method_title}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.legend(title="Emotion", loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_speaker_plot_3d(projection_df: pd.DataFrame, method_title: str, output_path: Path) -> None:
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    speaker_marker_map = build_speaker_marker_map(projection_df)

    for speaker_id in sorted(projection_df["speaker_id"].unique()):
        speaker_df = projection_df[projection_df["speaker_id"] == speaker_id]
        for emotion in EMOTION_ORDER:
            subset = speaker_df[speaker_df["emotion"] == emotion]
            if subset.empty:
                continue
            ax.scatter(
                subset["x"],
                subset["y"],
                subset["z"],
                s=28,
                alpha=0.68,
                color=EMOTION_PALETTE[emotion],
                marker=speaker_marker_map[speaker_id],
                depthshade=True,
            )

    add_emotion_centroids_3d(ax, projection_df)
    checkpoint_name = projection_df["checkpoint"].iloc[0]
    ax.set_title(f"Style Encoder Embedding Space by Speaker\n{checkpoint_name} | {method_title}")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")

    emotion_handles = [
        Line2D([0], [0], marker="o", color="w", label=emotion, markerfacecolor=EMOTION_PALETTE[emotion], markersize=8)
        for emotion in EMOTION_ORDER
    ]
    speaker_handles = [
        Line2D([0], [0], marker=marker, color="black", linestyle="", label=str(speaker), markersize=7)
        for speaker, marker in build_speaker_marker_map(projection_df).items()
    ]
    emotion_legend = ax.legend(handles=emotion_handles, title="Emotion", loc="upper left", frameon=True)
    ax.add_artist(emotion_legend)
    ax.legend(handles=speaker_handles, title="Speaker", loc="upper right", ncol=2, frameon=True)

    fig.tight_layout()
    fig.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(fig)


def save_emotion_plot_3d_plotly(projection_df: pd.DataFrame, method_title: str, output_path: Path) -> None:
    import plotly.express as px
    import plotly.graph_objects as go

    checkpoint_name = projection_df["checkpoint"].iloc[0]
    fig = px.scatter_3d(
        projection_df,
        x="x",
        y="y",
        z="z",
        color="emotion",
        color_discrete_map=EMOTION_PALETTE,
        category_orders={"emotion": EMOTION_ORDER},
        hover_data=["file_id", "speaker_id", "emotion", "wav_path"],
        title=f"Style Encoder Embedding Space by Emotion<br>{checkpoint_name} | {method_title}",
        opacity=0.72,
    )
    fig.update_traces(marker={"size": 4})

    centroids = get_emotion_centroids_3d(projection_df)
    fig.add_trace(
        go.Scatter3d(
            x=centroids["x"],
            y=centroids["y"],
            z=centroids["z"],
            mode="markers+text",
            text=centroids["emotion"],
            textposition="top center",
            marker={
                "size": 8,
                "symbol": "diamond",
                "color": [EMOTION_PALETTE[emotion] for emotion in centroids["emotion"]],
                "line": {"color": "black", "width": 1},
            },
            name="Emotion centroids",
            hoverinfo="skip",
        )
    )
    fig.update_layout(scene={"xaxis_title": "Component 1", "yaxis_title": "Component 2", "zaxis_title": "Component 3"})
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def save_speaker_plot_3d_plotly(projection_df: pd.DataFrame, method_title: str, output_path: Path) -> None:
    import plotly.express as px
    import plotly.graph_objects as go

    checkpoint_name = projection_df["checkpoint"].iloc[0]
    fig = px.scatter_3d(
        projection_df,
        x="x",
        y="y",
        z="z",
        color="emotion",
        symbol="speaker_id",
        color_discrete_map=EMOTION_PALETTE,
        category_orders={"emotion": EMOTION_ORDER},
        hover_data=["file_id", "speaker_id", "emotion", "wav_path"],
        title=f"Style Encoder Embedding Space by Speaker<br>{checkpoint_name} | {method_title}",
        opacity=0.7,
    )
    fig.update_traces(marker={"size": 4})

    centroids = get_emotion_centroids_3d(projection_df)
    fig.add_trace(
        go.Scatter3d(
            x=centroids["x"],
            y=centroids["y"],
            z=centroids["z"],
            mode="markers+text",
            text=centroids["emotion"],
            textposition="top center",
            marker={
                "size": 8,
                "symbol": "diamond",
                "color": [EMOTION_PALETTE[emotion] for emotion in centroids["emotion"]],
                "line": {"color": "black", "width": 1},
            },
            name="Emotion centroids",
            hoverinfo="skip",
        )
    )
    fig.update_layout(scene={"xaxis_title": "Component 1", "yaxis_title": "Component 2", "zaxis_title": "Component 3"})
    fig.write_html(str(output_path), include_plotlyjs="cdn")


def save_plotly_3d_outputs(projection_df: pd.DataFrame, method_title: str, output_dir: Path, method_name: str) -> bool:
    try:
        import plotly  # noqa: F401
    except Exception:
        return False

    save_emotion_plot_3d_plotly(projection_df, method_title, output_dir / f"style_space_{method_name}_by_emotion.html")
    save_speaker_plot_3d_plotly(projection_df, method_title, output_dir / f"style_space_{method_name}_by_speaker.html")
    return True


def save_projection_plots(projection_df: pd.DataFrame, method_title: str, output_dir: Path, method_name: str, dims: int) -> None:
    if dims == 2:
        save_emotion_plot_2d(projection_df, method_title, output_dir / f"style_space_{method_name}_by_emotion.png")
        save_speaker_plot_2d(projection_df, method_title, output_dir / f"style_space_{method_name}_by_speaker.png")
        return

    if save_plotly_3d_outputs(projection_df, method_title, output_dir, method_name):
        return

    print("Plotly is not installed. Falling back to static matplotlib 3D PNG outputs.")
    save_emotion_plot_3d(projection_df, method_title, output_dir / f"style_space_{method_name}_by_emotion.png")
    save_speaker_plot_3d(projection_df, method_title, output_dir / f"style_space_{method_name}_by_speaker.png")


def save_rank_summary(raw_df: pd.DataFrame, output_dir: Path) -> list[Path]:
    rank_columns = get_rank_columns(raw_df)
    rank_columns = [column for column in rank_columns if column != "rank_target"]
    if not rank_columns:
        return []

    produced: list[Path] = []
    long_rows = []
    for column in rank_columns:
        head_emotion = column[len(RANK_COLUMN_PREFIX):].capitalize()
        for row in raw_df[["file_id", "speaker_id", "emotion", column]].itertuples(index=False):
            long_rows.append(
                {
                    "file_id": row.file_id,
                    "speaker_id": row.speaker_id,
                    "emotion": row.emotion,
                    "rank_head": head_emotion,
                    "rank_score": float(getattr(row, column)),
                }
            )
    rank_long = pd.DataFrame(long_rows)
    rank_long_path = output_dir / "rank_scores_long.csv"
    rank_long.to_csv(rank_long_path, index=False)
    produced.append(rank_long_path)

    neutral_means = (
        rank_long[rank_long["emotion"] == "Neutral"]
        .groupby("rank_head")["rank_score"]
        .mean()
        .rename("neutral_mean")
    )
    summary = (
        rank_long.groupby(["rank_head", "emotion"])["rank_score"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .merge(neutral_means, on="rank_head", how="left")
    )
    summary["mean_minus_neutral"] = summary["mean"] - summary["neutral_mean"]
    summary_path = output_dir / "rank_summary_by_emotion.csv"
    summary.to_csv(summary_path, index=False)
    produced.append(summary_path)

    plot_df = rank_long[rank_long["rank_head"].isin([emotion for emotion in EMOTION_ORDER if emotion != "Neutral"])].copy()
    fig, axes = plt.subplots(2, 2, figsize=(16, 11), sharey=False)
    for ax, emotion in zip(axes.ravel(), [emotion for emotion in EMOTION_ORDER if emotion != "Neutral"]):
        subset = plot_df[plot_df["rank_head"] == emotion]
        draw_boxplot_by_emotion(ax, subset, "rank_score", EMOTION_ORDER)
        ax.set_title(f"Rank head: {emotion}")
        ax.set_xlabel("True emotion")
        ax.set_ylabel("Rank score")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    rank_plot_path = output_dir / "rank_scores_by_emotion.png"
    fig.savefig(rank_plot_path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    produced.append(rank_plot_path)

    if "rank_target" in raw_df.columns:
        fig, ax = plt.subplots(figsize=(11, 7))
        draw_boxplot_by_emotion(ax, raw_df, "rank_target", EMOTION_ORDER)
        ax.set_title("Rank Score from Each Sample's Own Emotion Head")
        ax.set_xlabel("True emotion")
        ax.set_ylabel("Target-head rank score")
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        target_plot_path = output_dir / "rank_target_by_emotion.png"
        fig.savefig(target_plot_path, dpi=240, bbox_inches="tight")
        plt.close(fig)
        produced.append(target_plot_path)

    return produced


def nearest_centroid_predictions(X: object, labels: object, label_order: list[str]) -> tuple[object, dict[str, object]]:
    centroids = {label: X[labels == label].mean(axis=0) for label in label_order}
    distances = [
        ((X - centroids[label]) ** 2).sum(axis=1) ** 0.5
        for label in label_order
    ]
    distance_matrix = np.column_stack(distances)
    predictions = np.array(label_order)[distance_matrix.argmin(axis=1)]
    return predictions, centroids


def compute_linear_cv_accuracy(X: object, labels: object, cv_splits: int, random_seed: int) -> float:
    min_class_count = pd.Series(labels).value_counts().min()
    if min_class_count < 2:
        return float("nan")
    splits = min(cv_splits, int(min_class_count))
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
    cv = StratifiedKFold(n_splits=splits, shuffle=True, random_state=random_seed)
    return float(cross_val_score(clf, X, labels, cv=cv, n_jobs=None).mean())


def save_separability_summary(raw_df: pd.DataFrame, output_dir: Path, random_seed: int, cv_splits: int = 5) -> list[Path]:
    embedding_columns = get_embedding_columns(raw_df)
    if not embedding_columns:
        return []

    produced: list[Path] = []
    X_raw = raw_df[embedding_columns].to_numpy(dtype=float)
    X = StandardScaler().fit_transform(X_raw)
    emotions = raw_df["emotion"].astype(str).to_numpy()
    speakers = raw_df["speaker_id"].astype(str).to_numpy()
    emotion_order = [emotion for emotion in EMOTION_ORDER if emotion in set(emotions)]
    speaker_order = sorted(pd.unique(raw_df["speaker_id"].astype(str)), key=lambda value: int(value) if str(value).isdigit() else str(value))

    emotion_sil_samples = silhouette_samples(X, emotions, metric="euclidean")
    emotion_pred, emotion_centroids = nearest_centroid_predictions(X, emotions, emotion_order)
    emotion_rows: list[dict] = [
        {
            "scope": "emotion_overall",
            "label": "ALL",
            "n": len(raw_df),
            "silhouette": float(silhouette_score(X, emotions, metric="euclidean")),
            "nearest_centroid_acc": float((emotion_pred == emotions).mean()),
            "linear_cv_acc": compute_linear_cv_accuracy(X_raw, emotions, cv_splits, random_seed),
            "within_radius": float("nan"),
            "nearest_label": "",
            "nearest_label_dist": float("nan"),
            "sep_ratio": float("nan"),
        }
    ]

    for emotion in emotion_order:
        mask = emotions == emotion
        own_distances = ((X[mask] - emotion_centroids[emotion]) ** 2).sum(axis=1) ** 0.5
        other_distances = {
            other: float((((emotion_centroids[emotion] - emotion_centroids[other]) ** 2).sum()) ** 0.5)
            for other in emotion_order
            if other != emotion
        }
        nearest_emotion = min(other_distances, key=other_distances.get)
        emotion_rows.append(
            {
                "scope": "emotion_class",
                "label": emotion,
                "n": int(mask.sum()),
                "silhouette": float(emotion_sil_samples[mask].mean()),
                "silhouette_p10": float(pd.Series(emotion_sil_samples[mask]).quantile(0.10)),
                "nearest_centroid_acc": float((emotion_pred[mask] == emotion).mean()),
                "linear_cv_acc": float("nan"),
                "within_radius": float(own_distances.mean()),
                "nearest_label": nearest_emotion,
                "nearest_label_dist": other_distances[nearest_emotion],
                "sep_ratio": float(other_distances[nearest_emotion] / own_distances.mean()),
            }
        )

    emotion_summary = pd.DataFrame(emotion_rows)
    emotion_summary_path = output_dir / "separability_emotion_summary.csv"
    emotion_summary.to_csv(emotion_summary_path, index=False)
    produced.append(emotion_summary_path)

    emotion_confusion = pd.DataFrame(
        confusion_matrix(emotions, emotion_pred, labels=emotion_order, normalize="true"),
        index=emotion_order,
        columns=emotion_order,
    )
    emotion_confusion.index.name = "true_emotion"
    emotion_confusion_path = output_dir / "separability_emotion_nearest_centroid_confusion.csv"
    emotion_confusion.to_csv(emotion_confusion_path)
    produced.append(emotion_confusion_path)

    speaker_summary_rows: list[dict] = []
    speaker_detail_rows: list[dict] = []
    for emotion in emotion_order:
        emotion_mask = emotions == emotion
        X_emotion = X[emotion_mask]
        X_raw_emotion = X_raw[emotion_mask]
        speakers_emotion = speakers[emotion_mask]
        local_speakers = [speaker for speaker in speaker_order if speaker in set(speakers_emotion)]
        if len(local_speakers) < 2:
            continue

        speaker_sil_samples = silhouette_samples(X_emotion, speakers_emotion, metric="euclidean")
        speaker_pred, speaker_centroids = nearest_centroid_predictions(X_emotion, speakers_emotion, local_speakers)
        speaker_summary_rows.append(
            {
                "emotion": emotion,
                "n": int(emotion_mask.sum()),
                "speaker_silhouette": float(silhouette_score(X_emotion, speakers_emotion, metric="euclidean")),
                "speaker_nearest_centroid_acc": float((speaker_pred == speakers_emotion).mean()),
                "speaker_linear_cv_acc": compute_linear_cv_accuracy(X_raw_emotion, speakers_emotion, cv_splits, random_seed),
            }
        )

        for speaker in local_speakers:
            speaker_mask = speakers_emotion == speaker
            own_distances = ((X_emotion[speaker_mask] - speaker_centroids[speaker]) ** 2).sum(axis=1) ** 0.5
            other_distances = {
                other: float((((speaker_centroids[speaker] - speaker_centroids[other]) ** 2).sum()) ** 0.5)
                for other in local_speakers
                if other != speaker
            }
            nearest_speaker = min(other_distances, key=other_distances.get)
            speaker_detail_rows.append(
                {
                    "emotion": emotion,
                    "speaker_id": speaker,
                    "n": int(speaker_mask.sum()),
                    "speaker_silhouette": float(speaker_sil_samples[speaker_mask].mean()),
                    "speaker_nearest_centroid_acc": float((speaker_pred[speaker_mask] == speaker).mean()),
                    "within_radius": float(own_distances.mean()),
                    "nearest_speaker": nearest_speaker,
                    "nearest_speaker_dist": other_distances[nearest_speaker],
                    "sep_ratio": float(other_distances[nearest_speaker] / own_distances.mean()),
                }
            )

    speaker_summary_path = output_dir / "separability_speaker_within_emotion_summary.csv"
    pd.DataFrame(speaker_summary_rows).to_csv(speaker_summary_path, index=False)
    produced.append(speaker_summary_path)

    speaker_detail_path = output_dir / "separability_speaker_within_emotion_per_speaker.csv"
    pd.DataFrame(speaker_detail_rows).to_csv(speaker_detail_path, index=False)
    produced.append(speaker_detail_path)

    return produced


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_output = repo_root / "experiments" / "style_embedding_viz"
    parser = argparse.ArgumentParser(
        description="Visualize the style encoder embedding space for a single checkpoint.",
        epilog=(
            "Example: python scripts/visualize_style_embeddings.py "
            "--config-path Configs/config_emo_balanced_holdout.yml "
            "--checkpoint-path Models/exp3_emo_balanced_holdout_from_scratch/epoch_00105.pth"
        ),
    )
    parser.add_argument("--config-path", required=True, help="Path to the model config YAML.")
    parser.add_argument("--checkpoint-path", required=True, help="Path to the model checkpoint.")
    parser.add_argument("--input-csv", default="Data/esd_balanced_train.csv", help="Metadata CSV with wav_path and emotion columns.")
    parser.add_argument("--output-dir", default=None, help=f"Directory for plots and CSVs. Defaults to {default_output}/<checkpoint_stem>.")
    parser.add_argument("--max-samples", type=int, default=None, help="Maximum number of samples to visualize.")
    parser.add_argument("--samples-per-emotion", type=int, default=None, help="Optional cap per emotion label.")
    parser.add_argument("--samples-per-speaker", type=int, default=None, help="Optional cap per speaker.")
    parser.add_argument("--methods", nargs="+", choices=["pca", "umap"], default=["pca", "umap"], help="Projection methods to compute.")
    parser.add_argument("--dims", nargs="+", type=int, choices=[2, 3], default=[2], help="Projection dimensionality to generate.")
    parser.add_argument("--random-seed", type=int, default=42, help="Random seed for sampling and UMAP.")
    parser.add_argument("--device", default=None, help="Torch device override, e.g. cuda or cpu.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for style encoder extraction.")
    parser.add_argument("--separability-cv-splits", type=int, default=5, help="Cross-validation folds for linear separability probes.")
    parser.add_argument("--skip-separability", action="store_true", help="Skip separability CSV summaries.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    for name in ["max_samples", "samples_per_emotion", "samples_per_speaker", "batch_size", "separability_cv_splits"]:
        value = getattr(args, name)
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be a positive integer when provided.")


def main() -> None:
    args = parse_args()
    validate_args(args)

    repo_root = Path(__file__).resolve().parent.parent
    config_path = resolve_path(args.config_path, repo_root)
    checkpoint_path = resolve_path(args.checkpoint_path, repo_root)
    input_csv = resolve_path(args.input_csv, repo_root)

    output_dir = Path(args.output_dir) if args.output_dir else repo_root / "experiments" / "style_embedding_viz" / checkpoint_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    if sns is not None:
        sns.set_theme(style="whitegrid", context="talk")
    else:
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "ggplot")

    input_df = load_input_frame(input_csv)
    sampled_df = sample_frame(
        input_df,
        max_samples=args.max_samples,
        samples_per_emotion=args.samples_per_emotion,
        samples_per_speaker=args.samples_per_speaker,
        random_seed=args.random_seed,
    )
    sampled_df["wav_path"] = sampled_df["wav_path"].map(lambda path: str(resolve_path(path, repo_root)))

    bundle = load_eval_bundle(str(config_path), str(checkpoint_path), device=args.device)
    bundle.checkpoint_name = checkpoint_path.name
    raw_df = extract_embeddings(bundle, sampled_df, batch_size=args.batch_size)

    raw_output_path = output_dir / "embeddings_raw.csv"
    raw_columns = ["file_id", "speaker_id", "emotion", "label", "wav_path", "checkpoint"] + get_embedding_columns(raw_df) + get_rank_columns(raw_df)
    raw_df = raw_df[raw_columns]
    raw_df.to_csv(raw_output_path, index=False)
    rank_outputs = save_rank_summary(raw_df, output_dir)
    separability_outputs = []
    if not args.skip_separability:
        separability_outputs = save_separability_summary(
            raw_df,
            output_dir,
            random_seed=args.random_seed,
            cv_splits=args.separability_cv_splits,
        )

    produced_methods: list[str] = []

    for dims in sorted(set(args.dims)):
        if "pca" in args.methods:
            projection_df, method_title = compute_pca_projection(raw_df, dims=dims)
            projection_suffix = "" if dims == 2 else f"_{dims}d"
            projection_path = output_dir / f"projection_pca{projection_suffix}.csv"
            projection_df.to_csv(projection_path, index=False)
            save_projection_plots(projection_df, method_title, output_dir, method_name=f"pca{projection_suffix}", dims=dims)
            produced_methods.append(f"pca-{dims}d")

        if "umap" in args.methods:
            umap_result = compute_umap_projection(raw_df, args.random_seed, dims=dims)
            if umap_result is None:
                print("Skipping UMAP because `umap-learn` is not usable in the active environment.")
            else:
                projection_df, method_title = umap_result
                projection_suffix = "" if dims == 2 else f"_{dims}d"
                projection_path = output_dir / f"projection_umap{projection_suffix}.csv"
                projection_df.to_csv(projection_path, index=False)
                save_projection_plots(projection_df, method_title, output_dir, method_name=f"umap{projection_suffix}", dims=dims)
                produced_methods.append(f"umap-{dims}d")

    if not produced_methods:
        raise RuntimeError("No projection outputs were produced. Check the requested methods and environment.")

    print(f"Saved raw embeddings to {raw_output_path}")
    print(f"Saved outputs to {output_dir}")
    if rank_outputs:
        print("Saved rank-head outputs:")
        for path in rank_outputs:
            print(f"  {path}")
    else:
        print("No rank-head was found in the loaded checkpoint/config; skipped rank-head outputs.")
    if separability_outputs:
        print("Saved separability outputs:")
        for path in separability_outputs:
            print(f"  {path}")
    print(f"Samples visualized: {len(raw_df)}")
    print(f"Methods completed: {', '.join(produced_methods)}")


if __name__ == "__main__":
    main()
