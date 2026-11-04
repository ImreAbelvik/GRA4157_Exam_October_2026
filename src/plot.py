
import matplotlib.pyplot as plt


def plot_learning_curve(results_df, laya_score, metric="accuracy",
                        title="Learning Curves: IMDb Sentiment"):

    plt.figure(figsize=(10, 6))

    colors = {
        "Logistic Regression": "blue",
        "Naive Bayes": "green"
    }

    for model in results_df["model"].unique():
        data = results_df[results_df["model"] == model]
        grouped = data.groupby("sample_size")[metric].agg(["mean", "std"])

        sizes = grouped.index
        means = grouped["mean"]
        stds = grouped["std"].fillna(0)

        plt.plot(sizes, means, marker="o", label=model,
                 color=colors.get(model))

        plt.fill_between(sizes, means - stds, means + stds,
                         alpha=0.2, color=colors.get(model))

    plt.axhline(laya_score, color="red", linestyle="--",
                label="Laya (Zero-Shot)")

    plt.xscale("log")
    plt.xlabel("Training Set Size")
    plt.ylabel(metric.capitalize())
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/plots/learning_curve_{metric}.pdf")
    plt.show()


def plot_efficiency(results_df, laya_infer_time):
    max_size = results_df["sample_size"].max()
    data = results_df[results_df["sample_size"] == max_size]

    grouped = data.groupby("model")["infer_time_per_1k_sec"].mean()

    models = grouped.index.tolist() + ["Laya (Zero-Shot)"]
    times = grouped.tolist() + [laya_infer_time]

    plt.figure(figsize=(8, 5))
    bars = plt.bar(models, times)

    plt.ylabel("Seconds per 1,000 Reviews")
    plt.title(f"Inference Time (Training Size: {max_size})")
    plt.grid(axis="y", linestyle="--", alpha=0.5)

    for bar in bars:
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}s",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()
    plt.savefig("results/plots/inference_efficiency.pdf")
    plt.show()
