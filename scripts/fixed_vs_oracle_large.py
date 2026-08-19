import os
import time
import h5py
import hnswlib
import numpy as np
import pandas as pd


# ============================================================
# 实验配置
# ============================================================

DATASETS = {
    "Cohere-1M": {
        "base": "../data/cohere_train.f32",
        "query": "../data/cohere_test.f32",
        "gt": "../data/cohere_groundtruth.i32",
        "dim": 768,
        "n_base": 1000000,
        "n_query": 1000,
        "gt_k": 1000,
        "space": "cosine"
    },
}

# TopK
TOPK = 10

# 单个 Query 的目标 Recall
TARGET_RECALL = 0.95

# 要求多少比例的 Query 达到目标 Recall
TARGET_COVERAGE = 0.90

# ============================================================
# 加速优化：减少候选ef，只测试关键值
# ============================================================

EF_CANDIDATES = [
    50, 100, 200, 400, 600, 800, 1000, 1200
]

ORACLE_EF_CANDIDATES = EF_CANDIDATES

# ============================================================
# 加速优化：降低构建参数
# ============================================================

HNSW_M = 12              # 从16降到12，构建速度提升约30%
HNSW_EF_CONSTRUCTION = 100  # 从200降到100，构建速度提升约50%

# 随机种子
RANDOM_SEED = 42

# 输出目录
RESULT_DIR = "../results"


# ============================================================
# 读取 Cohere 数据集 (.f32 / .i32 格式)
# ============================================================

def load_cohere_dataset(dataset_config):
    """加载 Cohere 数据集的 .f32 和 .i32 文件"""
    
    base_path = dataset_config["base"]
    query_path = dataset_config["query"]
    gt_path = dataset_config["gt"]
    dim = dataset_config["dim"]
    n_base = dataset_config["n_base"]
    n_query = dataset_config["n_query"]
    gt_k = dataset_config["gt_k"]
    
    print("\n" + "=" * 70)
    print("Loading Cohere dataset:")
    print(f"  Base: {base_path}")
    print(f"  Query: {query_path}")
    print(f"  GT: {gt_path}")
    print("=" * 70)
    
    # 检查文件是否存在
    for path in [base_path, query_path, gt_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
    
    # 加载 base vectors
    print("Loading base vectors...")
    base_data = np.fromfile(base_path, dtype=np.float32)
    train = base_data.reshape(n_base, dim)
    print(f"  Base vectors: {train.shape} ({train.nbytes / 1024**3:.2f} GiB)")
    
    # 加载 query vectors
    print("Loading query vectors...")
    query_data = np.fromfile(query_path, dtype=np.float32)
    queries = query_data.reshape(n_query, dim)
    print(f"  Query vectors: {queries.shape}")
    
    # 加载 ground truth
    print("Loading ground truth...")
    gt_data = np.fromfile(gt_path, dtype=np.int32)
    ground_truth = gt_data.reshape(n_query, gt_k)
    print(f"  Ground truth: {ground_truth.shape}")
    
    return train, queries, ground_truth


# ============================================================
# 判断距离类型
# ============================================================

def get_space(path):
    if isinstance(path, dict):
        return path.get("space", "cosine")
    
    filename = os.path.basename(path).lower()
    if "angular" in filename:
        return "cosine"
    elif "euclidean" in filename:
        return "l2"
    else:
        return "cosine"


# ============================================================
# 构建 HNSW（加速版）
# ============================================================

def build_index(base, space):
    dim = base.shape[1]

    print("\n" + "=" * 70)
    print("Building HNSW index (Fast Mode)...")
    print("=" * 70)
    print(f"  Dimension: {dim}")
    print(f"  Space: {space}")
    print(f"  M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}")
    print(f"  Total vectors: {len(base):,}")
    print("  Estimated time: 5-10 minutes")
    print()

    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(
        max_elements=len(base),
        ef_construction=HNSW_EF_CONSTRUCTION,
        M=HNSW_M,
        random_seed=RANDOM_SEED
    )
    
    # 分批添加并显示进度
    batch_size = 100000
    total = len(base)
    
    print("Adding vectors to index...")
    for i in range(0, total, batch_size):
        batch = base[i:min(i+batch_size, total)]
        index.add_items(batch)
        progress = min(100, int((i + batch_size) / total * 100))
        print(f"  Progress: {progress}% ({min(i+batch_size, total):,}/{total:,})", end="\r")
    
    print("\nIndex built!")
    return index


# ============================================================
# 计算每个 Query 的 Recall@K
# ============================================================

def calculate_query_recall(results, ground_truth, k):
    recalls = np.zeros(len(results), dtype=np.float32)
    for i in range(len(results)):
        retrieved = set(results[i][:k])
        gt = set(ground_truth[i][:k])
        recalls[i] = len(retrieved.intersection(gt)) / k
    return recalls


# ============================================================
# 运行一次固定 efSearch
# ============================================================

def run_fixed_search(index, queries, ground_truth, ef, k):
    index.set_ef(ef)

    start = time.perf_counter()
    labels, _ = index.knn_query(queries, k=k)
    elapsed = time.perf_counter() - start

    recalls = calculate_query_recall(labels, ground_truth, k)

    coverage = np.mean(recalls >= TARGET_RECALL)
    avg_recall = np.mean(recalls)
    latency_ms = elapsed / len(queries) * 1000
    qps = len(queries) / elapsed

    return {
        "ef": ef,
        "average_recall": avg_recall,
        "coverage": coverage,
        "latency_ms": latency_ms,
        "qps": qps,
        "recalls": recalls,
    }


# ============================================================
# 找到满足 Coverage 要求的最小 Fixed ef
# ============================================================

def find_fixed_baseline(index, queries, ground_truth, k):
    print("\n" + "=" * 70)
    print("Searching Fixed baseline")
    print("=" * 70)

    all_results = []

    for ef in EF_CANDIDATES:
        print(f"Testing Fixed ef={ef}...")

        result = run_fixed_search(index, queries, ground_truth, ef, k)

        print(
            f"  Recall={result['average_recall']:.4f} "
            f"Coverage={result['coverage']:.4%} "
            f"QPS={result['qps']:.2f}"
        )

        all_results.append(result)

        if result["coverage"] >= TARGET_COVERAGE:
            print(f"\n✓ Fixed baseline found: ef={ef}")
            return result, all_results

    print(f"\n⚠ WARNING: No Fixed efSearch can satisfy Coverage >= {TARGET_COVERAGE:.2%}")
    best = max(all_results, key=lambda x: x["coverage"])
    return best, all_results


# ============================================================
# Oracle
# ============================================================

def calculate_oracle(index, queries, ground_truth, k):
    print("\n" + "=" * 70)
    print("Calculating Oracle")
    print("=" * 70)

    num_queries = len(queries)
    optimal_ef = np.full(num_queries, np.nan, dtype=np.float32)

    for ef in ORACLE_EF_CANDIDATES:
        print(f"Oracle testing ef={ef}...")

        index.set_ef(ef)
        labels, _ = index.knn_query(queries, k=k)
        recalls = calculate_query_recall(labels, ground_truth, k)

        unfinished = np.isnan(optimal_ef)
        satisfied = recalls >= TARGET_RECALL
        update = unfinished & satisfied
        optimal_ef[update] = ef

        solved = np.sum(~np.isnan(optimal_ef))
        print(f"  Solved: {solved}/{num_queries} ({solved/num_queries*100:.1f}%)")

    valid = ~np.isnan(optimal_ef)
    oracle_coverage = np.mean(valid)

    if np.sum(valid) == 0:
        print("⚠ No query reaches target Recall.")
        return None

    oracle_average_ef = np.mean(optimal_ef[valid])
    oracle_median_ef = np.median(optimal_ef[valid])

    print("\nOracle result:")
    print(f"  Coverage = {oracle_coverage:.4%}")
    print(f"  Average optimal ef = {oracle_average_ef:.2f}")
    print(f"  Median optimal ef = {oracle_median_ef:.2f}")

    return {
        "optimal_ef": optimal_ef,
        "coverage": oracle_coverage,
        "average_ef": oracle_average_ef,
        "median_ef": oracle_median_ef,
        "valid": valid,
    }


# ============================================================
# 运行 Oracle 的实际搜索
# ============================================================

def run_oracle_search(index, queries, ground_truth, oracle, k):
    optimal_ef = oracle["optimal_ef"]
    valid = ~np.isnan(optimal_ef)
    valid_indices = np.where(valid)[0]

    if len(valid_indices) == 0:
        return None

    total_time = 0.0
    all_recalls = []

    print("\nRunning Oracle search...")
    print(f"  Querying {len(valid_indices)} queries with individual optimal ef")

    for idx in valid_indices:
        ef = int(optimal_ef[idx])
        index.set_ef(ef)

        start = time.perf_counter()
        labels, _ = index.knn_query(queries[idx:idx + 1], k=k)
        total_time += time.perf_counter() - start

        recall = calculate_query_recall(labels, ground_truth[idx:idx + 1], k)[0]
        all_recalls.append(recall)

    average_recall = np.mean(all_recalls)
    average_ef = np.mean(optimal_ef[valid])
    latency_ms = total_time / len(valid_indices) * 1000
    qps = len(valid_indices) / total_time

    return {
        "average_recall": average_recall,
        "coverage": oracle["coverage"],
        "average_ef": average_ef,
        "latency_ms": latency_ms,
        "qps": qps,
    }


# ============================================================
# 生成最终对比表
# ============================================================

def make_comparison_table(dataset_name, fixed, oracle_result):
    rows = []

    if fixed is not None:
        rows.append({
            "Dataset": dataset_name,
            "Method": "Fixed",
            "TopK": TOPK,
            "Target Recall": TARGET_RECALL,
            "Target Coverage": TARGET_COVERAGE,
            "Coverage": fixed["coverage"],
            "Average Recall": fixed["average_recall"],
            "Average efSearch": fixed["ef"],
            "Latency (ms)": fixed["latency_ms"],
            "QPS": fixed["qps"],
        })

    if oracle_result is not None:
        rows.append({
            "Dataset": dataset_name,
            "Method": "Oracle",
            "TopK": TOPK,
            "Target Recall": TARGET_RECALL,
            "Target Coverage": TARGET_COVERAGE,
            "Coverage": oracle_result["coverage"],
            "Average Recall": oracle_result["average_recall"],
            "Average efSearch": oracle_result["average_ef"],
            "Latency (ms)": oracle_result["latency_ms"],
            "QPS": oracle_result["qps"],
        })

    return rows


# ============================================================
# 单数据集实验
# ============================================================

def run_dataset(dataset_name, dataset_config):
    # 加载数据
    train, queries, ground_truth = load_cohere_dataset(dataset_config)

    space = get_space(dataset_config)
    index = build_index(train, space)

    # Fixed baseline
    fixed, fixed_all = find_fixed_baseline(index, queries, ground_truth, TOPK)

    # Oracle
    oracle = calculate_oracle(index, queries, ground_truth, TOPK)

    if oracle is not None:
        oracle_result = run_oracle_search(index, queries, ground_truth, oracle, TOPK)
    else:
        oracle_result = None

    rows = make_comparison_table(dataset_name, fixed, oracle_result)

    return rows, fixed_all, oracle


# ============================================================
# Main
# ============================================================

def main():
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("\n" + "=" * 80)
    print("FIXED vs ORACLE EXPERIMENT (Fast Mode)")
    print("=" * 80)
    print(f"TopK           = {TOPK}")
    print(f"Target Recall  = {TARGET_RECALL}")
    print(f"Target Coverage= {TARGET_COVERAGE}")
    print(f"HNSW: M={HNSW_M}, ef_construction={HNSW_EF_CONSTRUCTION}")

    all_rows = []

    for dataset_name, dataset_config in DATASETS.items():
        base_path = dataset_config["base"]
        if not os.path.exists(base_path):
            print(f"\n⚠ WARNING: {base_path} does not exist. Skipping.")
            continue

        rows, _, _ = run_dataset(dataset_name, dataset_config)
        all_rows.extend(rows)

    if len(all_rows) == 0:
        print("\nNo valid experiment results.")
        return

    df = pd.DataFrame(all_rows)

    output_path = os.path.join(RESULT_DIR, "fixed_vs_oracle_cohere.csv")
    df.to_csv(output_path, index=False)

    print("\n")
    print("=" * 100)
    print("FINAL FIXED vs ORACLE COMPARISON")
    print("=" * 100)
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
