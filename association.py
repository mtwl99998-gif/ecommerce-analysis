"""关联规则模块 —— 回答"哪些商品经常一起买？"

使用 Apriori 算法挖掘频繁项集，再提取关联规则。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules


def build_transaction_matrix(df, min_items=2):
    """将用户购买记录转为购物篮矩阵（one-hot 编码）。

    每个用户的所有购买商品构成一个"购物篮"。
    """
    buy_df = df[df["behavior_type"] == "buy"].copy()

    # 按用户聚合购买的商品
    baskets = buy_df.groupby("user_id")["item_id"].apply(list).reset_index()
    baskets = baskets[baskets["item_id"].apply(len) >= min_items]

    # 构建 one-hot 矩阵
    all_items = sorted(buy_df["item_id"].unique())
    item_to_col = {item: i for i, item in enumerate(all_items)}

    matrix = np.zeros((len(baskets), len(all_items)), dtype=bool)
    for i, items in enumerate(baskets["item_id"]):
        for item in items:
            matrix[i, item_to_col[item]] = True

    basket_matrix = pd.DataFrame(matrix, columns=all_items)
    basket_matrix.index = baskets["user_id"]

    return basket_matrix


def mine_rules(basket_matrix, min_support=0.02, min_confidence=0.3, metric="lift", min_threshold=1.0):
    """挖掘关联规则。

    Returns:
        rules_df sorted by lift descending
    """
    frequent_itemsets = apriori(
        basket_matrix, min_support=min_support, use_colnames=True, max_len=3
    )

    if frequent_itemsets.empty:
        print("未找到符合最小支持度的频繁项集，请降低 min_support")
        return pd.DataFrame()

    rules = association_rules(
        frequent_itemsets, metric=metric, min_threshold=min_threshold,
        num_itemsets=len(basket_matrix),
    )

    # 只保留 confidence >= min_confidence 的规则
    rules = rules[rules["confidence"] >= min_confidence]

    if not rules.empty:
        rules = rules.sort_values("lift", ascending=False)

    return rules


def enrich_rules_with_category(rules, item_meta):
    """给规则中的商品 ID 补充品类信息。"""
    if item_meta is None:
        return rules

    id_to_cat = dict(zip(item_meta["item_id"], item_meta["category"]))
    rules = rules.copy()

    def map_items(items):
        return ", ".join(f"{i}({id_to_cat.get(i, '?')})" for i in list(items)[:3])

    rules["antecedents_label"] = rules["antecedents"].apply(map_items)
    rules["consequents_label"] = rules["consequents"].apply(map_items)
    return rules


def top_rules_report(rules, top_n=15):
    """打印 Top N 关联规则报告。"""
    if rules.empty:
        print("无关联规则")
        return

    display_cols = ["antecedents_label", "consequents_label",
                    "support", "confidence", "lift"]
    available = [c for c in display_cols if c in rules.columns]
    top = rules[available].head(top_n)
    top_print = top.copy()
    for col in ["support", "confidence", "lift"]:
        if col in top_print.columns:
            top_print[col] = top_print[col].round(4)

    print(f"\nTop {top_n} 关联规则 (按 lift 排序)：")
    print(top_print.to_string(index=False))
    return top_print


def plot_rules(rules, save_path=None):
    """可视化关联规则：support vs confidence vs lift。"""
    if rules.empty:
        print("无关联规则可绘制")
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Support vs Confidence，颜色表示 Lift
    sample = rules.sample(min(100, len(rules)), random_state=42)
    sc = axes[0].scatter(
        sample["support"], sample["confidence"],
        c=sample["lift"], cmap="viridis", s=80,
        alpha=0.7, edgecolors="gray", linewidth=0.3,
    )
    axes[0].set_xlabel("Support")
    axes[0].set_ylabel("Confidence")
    axes[0].set_title("关联规则分布（颜色=Lift）", fontsize=13, fontweight="bold")
    plt.colorbar(sc, ax=axes[0], label="Lift")

    # Top 10 规则 Lift 柱状图
    top10 = rules.head(10).copy()
    top10["rule_label"] = top10.apply(
        lambda r: f"{list(r['antecedents'])} → {list(r['consequents'])}", axis=1
    )
    colors = ["#e74c3c" if l > 3 else "#f39c12" if l > 1.5 else "#3498db"
              for l in top10["lift"]]
    axes[1].barh(range(len(top10)), top10["lift"], color=colors)
    axes[1].set_yticks(range(len(top10)))
    axes[1].set_yticklabels(top10["rule_label"], fontsize=8)
    axes[1].set_xlabel("Lift")
    axes[1].set_title("Top 10 关联规则 Lift 值", fontsize=13, fontweight="bold")
    axes[1].invert_yaxis()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    return fig
