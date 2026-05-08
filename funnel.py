"""漏斗分析模块 —— 回答"用户在哪一步流失了？"

核心分析：
1. 整体转化漏斗（pv → fav → cart → buy）
2. 各品类漏斗对比
3. 时间维度漏斗趋势
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


BEHAVIOR_PRIORITY = {"pv": 1, "fav": 2, "cart": 3, "buy": 4}


def overall_funnel(df):
    """计算整体转化漏斗。

    Returns:
        DataFrame: stage, stage_cn, users, overall_rate_pct, step_rate_pct
    """
    user_stage = (
        df.groupby("user_id")["behavior_type"]
        .apply(lambda x: x.map(BEHAVIOR_PRIORITY).max())
        .reset_index()
    )
    user_stage["stage"] = user_stage["behavior_type"].map(
        {v: k for k, v in BEHAVIOR_PRIORITY.items()}
    )

    stage_order = ["pv", "fav", "cart", "buy"]
    funnel_data = []
    prev = None

    for stage in stage_order:
        count = (user_stage["behavior_type"] >= BEHAVIOR_PRIORITY[stage]).sum()
        overall_rate = count / len(user_stage) * 100
        step_rate = (count / prev * 100) if prev else 100.0
        funnel_data.append({
            "stage": stage,
            "stage_cn": {"pv": "浏览", "fav": "收藏", "cart": "加购", "buy": "购买"}[stage],
            "users": count,
            "overall_rate_pct": round(overall_rate, 2),
            "step_rate_pct": round(step_rate, 2),
        })
        prev = count

    return pd.DataFrame(funnel_data)


def funnel_by_category(df, item_meta):
    """按商品品类计算转化漏斗。"""
    if item_meta is None:
        raise ValueError("需要 item_meta 才能按品类分析")

    df_merged = df.merge(item_meta, on="item_id", how="left")
    user_cat_stage = (
        df_merged.groupby(["user_id", "category"])["behavior_type"]
        .apply(lambda x: x.map(BEHAVIOR_PRIORITY).max())
        .reset_index()
    )
    user_cat_stage["stage"] = user_cat_stage["behavior_type"].map(
        {v: k for k, v in BEHAVIOR_PRIORITY.items()}
    )

    results = []
    stage_order = ["pv", "fav", "cart", "buy"]
    for category in user_cat_stage["category"].unique():
        subset = user_cat_stage[user_cat_stage["category"] == category]
        total = len(subset)
        prev = None
        for stage in stage_order:
            count = (subset["behavior_type"] >= BEHAVIOR_PRIORITY[stage]).sum()
            results.append({
                "category": category,
                "stage": stage,
                "users": count,
                "overall_rate_pct": round(count / total * 100, 2),
                "step_rate_pct": round(count / prev * 100, 2) if prev else 100.0,
            })
            prev = count

    return pd.DataFrame(results)


def plot_funnel(funnel_df, title="用户行为转化漏斗", save_path=None):
    """绘制漏斗图（用户数 + 转化率双图）。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    stages = funnel_df["stage_cn"].tolist()
    users = funnel_df["users"].tolist()
    rates = funnel_df["step_rate_pct"].tolist()

    colors = ["#2ecc71", "#3498db", "#f39c12", "#e74c3c"]

    # 左：用户数
    bars = axes[0].bar(stages, users, color=colors, width=0.5, edgecolor="white")
    axes[0].set_title("各阶段用户数", fontsize=13, fontweight="bold")
    axes[0].set_ylabel("用户数")
    for bar, val in zip(bars, users):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(users) * 0.02,
                     f"{val:,}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    # 右：转化率
    bar_colors = ["#3498db"] + [
        "#27ae60" if r > 30 else "#e67e22" if r > 10 else "#e74c3c" for r in rates[1:]
    ]
    bars = axes[1].bar(stages, rates, color=bar_colors, width=0.5, edgecolor="white")
    axes[1].set_title("逐级转化率 (%)", fontsize=13, fontweight="bold")
    axes[1].set_ylabel("转化率 %")
    axes[1].axhline(y=20, color="gray", linestyle="--", alpha=0.5, label="20% 参考线")
    for bar, val in zip(bars, rates):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)

    fig.suptitle(title, fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"漏斗图已保存至 {save_path}")

    plt.show()
    return fig


def daily_funnel_trend(df):
    """计算每日漏斗转化率趋势。"""
    df = df.copy()
    df["date"] = df["timestamp"].dt.date

    daily = df.groupby(["date", "user_id"])["behavior_type"].apply(
        lambda x: x.map(BEHAVIOR_PRIORITY).max()
    ).reset_index()
    daily["stage"] = daily["behavior_type"].map(
        {v: k for k, v in BEHAVIOR_PRIORITY.items()}
    )

    trends = []
    for date, group in daily.groupby("date"):
        total = len(group)
        row = {"date": date, "total_users": total}
        for stage in ["pv", "fav", "cart", "buy"]:
            cnt = (group["behavior_type"] >= BEHAVIOR_PRIORITY[stage]).sum()
            row[f"{stage}_rate"] = round(cnt / total * 100, 2)
        trends.append(row)

    return pd.DataFrame(trends).sort_values("date")
