"""RFM 用户分层模块 —— 回答"高价值用户是谁？流失用户是谁？"

R (Recency)：最近一次消费距今天数 → 越小越好
F (Frequency)：消费频次 → 越大越好
M (Monetary)：消费金额 → 越大越好
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_rfm(df, item_meta=None, reference_date=None):
    """从行为数据中提取购买行为，计算每个用户的 R/F/M 值。

    Args:
        df: 行为数据
        item_meta: 商品元数据（含价格），无则为 None，M 值用购买次数替代
        reference_date: 参考日期，默认为数据中最后一天 + 1

    Returns:
        rfm_df: user_id, recency, frequency, monetary, R, F, M, RFM_score
    """
    buy_df = df[df["behavior_type"] == "buy"].copy()
    if buy_df.empty:
        raise ValueError("数据中没有购买行为，无法计算 RFM")

    if reference_date is None:
        reference_date = buy_df["timestamp"].max() + pd.Timedelta(days=1)

    # 合并商品价格（用于 M 值）
    if item_meta is not None and "price" in item_meta.columns:
        buy_df = buy_df.merge(item_meta[["item_id", "price"]], on="item_id", how="left")
        rfm = buy_df.groupby("user_id").agg(
            recency=("timestamp", lambda x: (reference_date - x.max()).days),
            frequency=("behavior_type", "count"),
            monetary=("price", "sum"),
        ).reset_index()
    else:
        rfm = buy_df.groupby("user_id").agg(
            recency=("timestamp", lambda x: (reference_date - x.max()).days),
            frequency=("behavior_type", "count"),
        ).reset_index()
        rfm["monetary"] = rfm["frequency"]  # 无价格时用频次代替

    rfm["monetary"] = rfm["monetary"].round(2)
    return rfm


def score_rfm(rfm_df):
    """对 R/F/M 分别打分（1-4 分），分数越高越好。

    R: 距离越近分数越高（反向打分）
    F: 频次越高分数越高
    M: 金额越高分数越高
    """
    df = rfm_df.copy()

    # Recency：反向打分（天数越少分越高）
    df["R"] = pd.qcut(df["recency"].rank(method="first"), q=4, labels=[4, 3, 2, 1])
    # Frequency：正向打分
    df["F"] = pd.qcut(df["frequency"].rank(method="first"), q=4, labels=[1, 2, 3, 4])
    # Monetary：正向打分
    df["M"] = pd.qcut(df["monetary"].rank(method="first"), q=4, labels=[1, 2, 3, 4])

    for col in ["R", "F", "M"]:
        df[col] = df[col].astype(int)

    df["RFM_score"] = df["R"] * 100 + df["F"] * 10 + df["M"]
    return df


def label_users(rfm_scored):
    """根据 R/F/M 分值给用户打标签。

    标签规则：
    - 重要价值用户：R>=3, F>=3, M>=3
    - 重要发展用户：R>=3, F>=3, M<3
    - 重要保持用户：R<3, F>=3, M>=3
    - 重要挽留用户：R<3, F<3, M>=3
    - 一般价值用户：R>=3, F<3, M>=3
    - 一般发展用户：R>=3, F<3, M<3 或 R>=3, F>=3, M<3 已覆盖
    - 一般保持用户：R<3, F>=3, M<3
    - 一般挽留用户：R<3, F<3, M<3
    """
    df = rfm_scored.copy()
    conditions = [
        (df["R"] >= 3) & (df["F"] >= 3) & (df["M"] >= 3),
        (df["R"] >= 3) & (df["F"] >= 3) & (df["M"] < 3),
        (df["R"] < 3)  & (df["F"] >= 3) & (df["M"] >= 3),
        (df["R"] < 3)  & (df["F"] < 3)  & (df["M"] >= 3),
        (df["R"] >= 3) & (df["F"] < 3)  & (df["M"] >= 3),
        (df["R"] < 3)  & (df["F"] >= 3) & (df["M"] < 3),
    ]
    labels = [
        "重要价值用户", "重要发展用户", "重要保持用户",
        "重要挽留用户", "一般价值用户", "一般保持用户",
    ]
    df["user_label"] = np.select(conditions, labels, default="一般挽留用户")
    return df


def rfm_summary(rfm_labeled):
    """输出 RFM 分层汇总表。"""
    summary = (
        rfm_labeled.groupby("user_label")
        .agg(
            用户数=("user_id", "count"),
            占比=("user_id", lambda x: round(len(x) / len(rfm_labeled) * 100, 1)),
            平均R=("R", "mean"),
            平均F=("F", "mean"),
            平均M=("M", "mean"),
        )
        .sort_values("用户数", ascending=False)
    )
    summary["平均R"] = summary["平均R"].round(1)
    summary["平均F"] = summary["平均F"].round(1)
    summary["平均M"] = summary["平均M"].round(1)
    return summary


def plot_rfm(rfm_labeled, save_path=None):
    """可视化 RFM 分层结果。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # 用户标签分布（饼图）
    label_counts = rfm_labeled["user_label"].value_counts()
    colors = ["#27ae60", "#2ecc71", "#3498db", "#f39c12", "#e67e22", "#e74c3c"]
    wedges, texts, autotexts = axes[0].pie(
        label_counts.values, labels=label_counts.index,
        autopct="%1.1f%%", colors=colors[:len(label_counts)],
        startangle=90, textprops={"fontsize": 8},
    )
    axes[0].set_title("用户分层占比", fontsize=13, fontweight="bold")

    # R/F/M 均值对比
    label_means = (
        rfm_labeled.groupby("user_label")[["R", "F", "M"]]
        .mean()
        .reindex(label_counts.index)
    )
    label_means.plot(kind="bar", ax=axes[1], color=["#e74c3c", "#3498db", "#2ecc71"])
    axes[1].set_title("各层用户 R/F/M 均值", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("平均分值 (1-4)")
    axes[1].legend(fontsize=9)
    axes[1].tick_params(axis="x", rotation=30, labelsize=8)

    # R vs F vs M 散点矩阵
    sample = rfm_labeled.sample(min(500, len(rfm_labeled)), random_state=42)
    axes[2].scatter(
        sample["R"], sample["F"],
        s=sample["M"] * 20, alpha=0.5,
        c=sample["R"] + sample["F"] + sample["M"],
        cmap="RdYlGn", edgecolors="gray", linewidth=0.3,
    )
    axes[2].set_xlabel("Recency 分值")
    axes[2].set_ylabel("Frequency 分值")
    axes[2].set_title("用户 R-F 分布（气泡大小=M值）", fontsize=13, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
