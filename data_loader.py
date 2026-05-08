"""数据加载与预处理模块。

支持两种模式：
1. 从 CSV 加载真实数据
2. 自动生成模拟数据（用于学习和演示）
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta


def generate_simulated_data(n_users=5000, n_items=200, seed=42):
    """生成模拟电商行为数据，用于没有真实数据时的学习和演示。

    生成四类行为：pv（浏览）、fav（收藏）、cart（加购）、buy（购买）
    行为之间存在漏斗关系，模拟真实用户行为分布。
    """
    rng = np.random.default_rng(seed)

    user_ids = np.arange(1, n_users + 1)
    item_ids = np.arange(1, n_items + 1)
    categories = rng.choice(
        ["电子产品", "服装", "食品", "家居", "美妆", "图书"],
        size=n_items,
        p=[0.15, 0.25, 0.20, 0.18, 0.12, 0.10],
    )

    item_meta = pd.DataFrame({
        "item_id": item_ids,
        "category": categories,
        "price": np.round(rng.uniform(10, 2000, n_items), 2),
    })

    records = []
    start_date = datetime(2026, 3, 1)
    end_date = datetime(2026, 4, 30)

    for user_id in user_ids:
        activity_level = rng.exponential(scale=2.0)  # 大部分用户 1-10 次会话
        n_sessions = max(1, int(activity_level) + 1)

        for _ in range(n_sessions):
            session_date = start_date + timedelta(
                days=int(rng.integers(0, (end_date - start_date).days))
            )
            session_hour = rng.choice(np.arange(24), p=_hour_distribution())
            ts = session_date + timedelta(
                hours=int(session_hour), minutes=int(rng.integers(0, 60))
            )

            viewed_items = rng.choice(item_ids, size=rng.integers(3, 16), replace=False)

            for i, item_id in enumerate(viewed_items):
                records.append({
                    "user_id": user_id, "item_id": item_id,
                    "behavior_type": "pv",
                    "timestamp": ts + timedelta(seconds=i * 30),
                })
                if rng.random() < 0.12:
                    records.append({
                        "user_id": user_id, "item_id": item_id,
                        "behavior_type": "fav",
                        "timestamp": ts + timedelta(seconds=i * 30 + 10),
                    })
                if rng.random() < 0.08:
                    records.append({
                        "user_id": user_id, "item_id": item_id,
                        "behavior_type": "cart",
                        "timestamp": ts + timedelta(seconds=i * 30 + 15),
                    })
                if rng.random() < 0.03:
                    records.append({
                        "user_id": user_id, "item_id": item_id,
                        "behavior_type": "buy",
                        "timestamp": ts + timedelta(seconds=i * 30 + 20),
                    })

    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["behavior_type"] = df["behavior_type"].astype("category")
    return df, item_meta


def _hour_distribution():
    """模拟一天中各小时的活跃度分布（双峰：午间 + 晚间）。"""
    hours = np.arange(24)
    weights = np.ones(24) * 0.02
    weights[10:14] = 0.06
    weights[19:24] = 0.08
    weights[20:23] = 0.10
    return weights / weights.sum()


def _find_data_file(data_dir="data"):
    """在 data/ 目录下自动查找 CSV，排除 item_meta.csv。"""
    data_path = Path(data_dir)
    if not data_path.exists():
        return None, None

    csv_files = [f for f in data_path.glob("*.csv") if f.name != "item_meta.csv"]
    if not csv_files:
        return None, None

    data_file = csv_files[0]
    item_meta_file = data_path / "item_meta.csv"
    return str(data_file), str(item_meta_file) if item_meta_file.exists() else None


def load_data(data_path=None, use_simulated=True, n_users=5000):
    """加载数据的统一入口。

    优先级：
    1. 指定了 data_path → 直接加载
    2. data/ 目录有 CSV → 自动加载（无需手动指定）
    3. 都没有 → 生成模拟数据

    Args:
        data_path: CSV 文件路径（可选，会自动检测 data/ 目录）
        use_simulated: 没有数据时是否生成模拟数据
        n_users: 模拟数据的用户数

    Returns:
        (behavior_df, item_meta_df)
    """
    _auto_meta = None  # 自动检测到的 item_meta 路径

    if not data_path:
        auto_path, _auto_meta = _find_data_file()
        if auto_path:
            data_path = auto_path
            print(f"自动检测到数据文件：{auto_path}")

    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path, parse_dates=["timestamp"])
        meta_path = _auto_meta or (Path(data_path).parent / "item_meta.csv")
        item_meta = pd.read_csv(meta_path) if Path(meta_path).exists() else None
        print(f"已加载真实数据：{len(df):,} 条行为记录")
        return df, item_meta

    if use_simulated:
        print(f"生成模拟数据：{n_users} 个用户...")
        df, item_meta = generate_simulated_data(n_users=n_users)
        print(f"共 {len(df):,} 条行为记录，{item_meta['category'].nunique()} 个品类")
        return df, item_meta

    raise FileNotFoundError("未提供数据路径且未启用模拟数据")


def clean_data(df):
    """基础数据清洗。"""
    df = df.copy()
    before = len(df)
    df = df.dropna(subset=["user_id", "item_id", "behavior_type"])
    if before != len(df):
        print(f"  去除缺失值：{before - len(df)} 行")

    dup_cols = ["user_id", "item_id", "behavior_type", "timestamp"]
    before = len(df)
    df = df.drop_duplicates(subset=dup_cols)
    if before != len(df):
        print(f"  去除重复：{before - len(df)} 行")

    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    behavior_map = {
        "pv": "pv", "page_view": "pv", "view": "pv",
        "fav": "fav", "favorite": "fav", "collect": "fav",
        "cart": "cart", "add_to_cart": "cart",
        "buy": "buy", "purchase": "buy", "order": "buy",
    }
    df["behavior_type"] = (
        df["behavior_type"].str.lower().map(behavior_map).fillna(df["behavior_type"])
    )
    return df
