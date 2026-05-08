# 电商用户行为分析项目


## 项目概述

基于电商平台用户行为数据（浏览/收藏/加购/购买），完成三个核心分析：

| 模块 | 核心问题 | 分析方法 |
|------|---------|---------|
| 漏斗分析 | 用户在哪一步流失？ | 整体漏斗 + 品类对比 + 时间趋势 |
| RFM 分层 | 高价值用户是谁？ | R/F/M 打分 + 8 层用户标签 |
| 关联规则 | 哪些商品经常一起买？ | Apriori 算法 + Lift 排序 |

## 项目结构

```
ecommerce-analysis/
├── data/               # 数据目录（放 CSV）
├── src/                # Python 源码
│   ├── data_loader.py  # 数据加载/清洗/模拟数据生成
│   ├── funnel.py       # 漏斗分析
│   ├── rfm.py          # RFM 用户分层
│   └── association.py  # 关联规则挖掘
├── notebooks/          # Jupyter Notebooks（按顺序执行）
│   ├── 01_data_overview.ipynb
│   ├── 02_funnel_analysis.ipynb
│   ├── 03_rfm_analysis.ipynb
│   └── 04_association_rules.ipynb
├── output/             # 输出图表/CSV
├── reports/            # 分析报告
└── requirements.txt
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动 Jupyter
jupyter notebook notebooks/
```

## 数据说明

项目内置模拟数据生成器，无需下载任何数据即可运行。

有真实数据时，将 CSV 放入 `data/` 目录，修改 notebook 中的 `load_data(data_path='data/your_file.csv')` 即可。数据需包含以下列：

| 列名 | 说明 | 示例 |
|------|------|------|
| user_id | 用户 ID | 1001 |
| item_id | 商品 ID | 20001 |
| behavior_type | 行为类型 | pv / fav / cart / buy |
| timestamp | 时间戳 | 2026-03-15 14:30:00 |

可选的 `item_meta.csv` 放商品信息（item_id, category, price）。
