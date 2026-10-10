from pathlib import Path
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd

def compare_variants(exp_dir:str):
    exp_path = Path(exp_dir)
    # check inside exp_dir for any folders that contains the name full_eval_{epoch_num}
    eval_dirs = list(exp_path.glob("full_eval_*"))
    for eval_dir in tqdm(eval_dirs):
        if eval_dir.is_dir():
            metrics_file = eval_dir / "overall_metrics.csv"
            if metrics_file.exists():
                print(f"Found metrics file: {metrics_file}")
                df = pd.read_csv(metrics_file) 
                #concat df with the epoch number extracted from the eval_dir name
                epoch_num = eval_dir.name.split("_")[3] 
                df["epoch"] = epoch_num
                # concat all dfs together
                if "all_metrics_df" not in locals():
                    all_metrics_df = df
                else:
                    all_metrics_df = pd.concat([all_metrics_df, df], ignore_index=True)

    all_metrics_df.to_csv(exp_path / "overall_metrics_comparison.csv", index=False)
    print(f"saved combined metrics to: {exp_path / 'overall_metrics_comparison.csv'}")
    summary_df = (
        all_metrics_df.groupby("epoch")[["emotion_acc", "emotion_ref_cosine", "speaker_cosine"]]
        .mean()
        .reset_index()
    )
    summary_df["epoch"] = pd.to_numeric(summary_df["epoch"], errors="coerce")
    summary_df = summary_df.sort_values("epoch")

    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in ["emotion_acc", "emotion_ref_cosine", "speaker_cosine"]:
        ax.plot(summary_df["epoch"], summary_df[metric], marker="o", label=metric)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Metric value")
    ax.set_xticks(range(0, 105, 5))
    ax.set_title("Model performance across epochs")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    plt.savefig(exp_path / "metrics_comparison.png")
    print(f"Saved metrics comparison plot to: {exp_path / 'metrics_comparison.png'}")
    plt.show()

    return all_metrics_df


def main():
    parser = argparse.ArgumentParser(description="Compare evaluation metrics across different epochs.")
    parser.add_argument("--exp-dir", type=str, help="Path to the experiments directory containing evaluation results.", default='experiments')
    parser.add_argument("--out-dir", type=str, help="Path to save the comparison results and plots.", default='experiments')
    args = parser.parse_args()
    compare_variants(args.exp_dir)
    
if __name__ == "__main__":
    main()
