# VSAG 支持动态搜索能力（Adaptive efSearch）

项目申请书：《孙飞宇-26c520014-项目申请书.md》
项目编号：26c520014 ｜ 申请人：孙飞宇 ｜ 导师：蒋超

本项目研究 **Query-level Adaptive efSearch**：不同 Query 的搜索难度不同，简单 Query 只需较小的 efSearch 即可达到目标召回率，困难 Query 则需要更大的搜索范围。固定 efSearch 会同时造成简单 Query 的 **Oversearch** 与困难 Query 的 **Undersearch**。项目通过在 HNSWLIB 上验证自适应搜索原型，最终将方案集成到 VSAG HGraph。

## 仓库结构

```
├── README.md                 # 本文件：项目说明
├── data/
│   └── README.md             # 数据集说明（数据文件过大，未随仓库上传）
├── scripts/                  # 实验代码
│   ├── adaptive_result_demo.py   # 2.5 方案初步效果呈现的代码
│   ├── fixed_vs_oracle_large.py  # 第一章结果对比的代码（Cohere-1M）
│   └── fixed_vs_oracle.py        # 第一章对比其他数据集的代码
└── results/                  # 数据输出结果
    ├── adaptive_result_demo.txt
    ├── fixed_vs_oracle.txt
    ├── fixed_vs_oracle_coco.csv
    ├── fixed_vs_oracle_fashion-mnist.csv
    ├── fixed_vs_oracle_nytimes.csv
    └── fixed_vs_oracle_shift-128.csv
```

## 脚本与申请书章节对应关系

| 脚本 | 对应章节 | 说明 |
|------|---------|------|
| `scripts/fixed_vs_oracle_large.py` | 第 1 章「项目背景」 | 在 Cohere-1M-wikipedia-768d 数据集上对比 Fixed efSearch 基线（ef=1200，Coverage 88.5%，QPS 644.71）与 Oracle 理论最优（平均 ef 121.19，QPS 1414.54，提升 119.4%） |
| `scripts/fixed_vs_oracle.py` | 第 1 章「对比其他数据集」 | 在 COCO-T2I、SIFT-128、NYTimes-256、Fashion-MNIST 四个数据集上验证不同 Query 搜索深度的差异（如 SIFT 理论搜索量可降低约 62.52%） |
| `scripts/adaptive_result_demo.py` | 2.5「方案初步效果呈现」 | 两阶段自适应搜索原型：第一阶段 ef=50/100/150 快速搜索，召回率 > 0.9 的查询提前停止，仅对"困难"查询使用 ef=400/800 精确搜索。相比 Fixed ef=200，QPS 提升 10%（2255.83 → 2489.03），平均 ef 从 200 降至 128.50 |

## 运行方式

1. 安装依赖：`pip install hnswlib h5py numpy pandas`
2. 按 [`data/README.md`](data/README.md) 下载数据集放入 `data/` 目录（保持文件名不变）
3. 在 `scripts/` 目录下运行对应脚本，输出写入 `results/`

## 技术方案

两阶段自适应搜索机制（2.3 节）：

1. **搜索前预测**：根据 Query 难度特征（距离分布、Top-K 距离 Gap、局部图密度等）预测初始 efSearch；
2. **搜索中动态控制**：持续监控候选集与 Top-K 结果状态，稳定则提前终止（Early Termination），不足则复用搜索状态动态增加 ef（50 → 100 → 150 → …）。

与单独采用 Ada-ef 或 DARTH 相比，本方案兼顾搜索宽度与搜索过程两个层面的优化。最终验收链路：**Static efSearch → Oracle 分析 → HNSWLIB Adaptive 原型 → VSAG HGraph 集成验证**（2.6 节）。
