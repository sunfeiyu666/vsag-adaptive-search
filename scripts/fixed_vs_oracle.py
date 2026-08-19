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
    "coco-t2i": "../data/coco-t2i-512-angular.hdf5",
}

# TopK
TOPK = 10

# 单个 Query 的目标 Recall
TARGET_RECALL = 0.95

# 要求多少比例的 Query 达到目标 Recall
TARGET_COVERAGE = 0.90

# Fixed efSearch 候选值
EF_CANDIDATES = [
    10,
    20,
    50,
    100,
    200,
    300,
    500,
    800,
    1000,
]

# Oracle 搜索候选值
ORACLE_EF_CANDIDATES = EF_CANDIDATES

# 随机种子
RANDOM_SEED = 42

# 输出目录
RESULT_DIR = "../results"


# ============================================================
# 读取 HDF5 数据集
# ============================================================

def load_dataset(path):

    print("\n" + "=" * 70)
    print("Loading dataset:")
    print(path)
    print("=" * 70)

    with h5py.File(path, "r") as f:

        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"])

    print("Base vectors:", train.shape)
    print("Query vectors:", test.shape)
    print("Ground truth:", neighbors.shape)

    return train, test, neighbors


# ============================================================
# 判断距离类型
# ============================================================

def get_space(path):

    filename = os.path.basename(path).lower()

    if "angular" in filename:
        return "cosine"

    elif "euclidean" in filename:
        return "l2"

    else:
        raise ValueError(
            f"Cannot determine metric from filename: {filename}"
        )


# ============================================================
# 构建 HNSW
# ============================================================

def build_index(base, space):

    dim = base.shape[1]

    print("\nBuilding HNSW index...")
    print("Dimension:", dim)
    print("Space:", space)

    index = hnswlib.Index(
        space=space,
        dim=dim
    )

    index.init_index(
        max_elements=len(base),
        ef_construction=200,
        M=16,
        random_seed=RANDOM_SEED
    )

    index.add_items(base)

    print("Index built")

    return index


# ============================================================
# 计算每个 Query 的 Recall@K
# ============================================================

def calculate_query_recall(results, ground_truth, k):

    recalls = np.zeros(
        len(results),
        dtype=np.float32
    )

    for i in range(len(results)):

        retrieved = set(results[i][:k])

        gt = set(ground_truth[i][:k])

        recalls[i] = len(
            retrieved.intersection(gt)
        ) / k

    return recalls


# ============================================================
# 运行一次固定 efSearch
# ============================================================

def run_fixed_search(
    index,
    queries,
    ground_truth,
    ef,
    k
):

    index.set_ef(ef)

    start = time.perf_counter()

    labels, _ = index.knn_query(
        queries,
        k=k
    )

    elapsed = time.perf_counter() - start

    recalls = calculate_query_recall(
        labels,
        ground_truth,
        k
    )

    coverage = np.mean(
        recalls >= TARGET_RECALL
    )

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

def find_fixed_baseline(
    index,
    queries,
    ground_truth,
    k
):

    print("\n" + "=" * 70)
    print("Searching Fixed baseline")
    print("=" * 70)

    all_results = []

    for ef in EF_CANDIDATES:

        print(f"Testing Fixed ef={ef}")

        result = run_fixed_search(
            index,
            queries,
            ground_truth,
            ef,
            k
        )

        print(
            f"ef={ef:<4d} "
            f"Recall={result['average_recall']:.4f} "
            f"Coverage={result['coverage']:.4%} "
            f"Latency={result['latency_ms']:.4f} ms "
            f"QPS={result['qps']:.2f}"
        )

        all_results.append(result)

        # 找到满足 Coverage 的最小 ef
        if result["coverage"] >= TARGET_COVERAGE:

            print(
                f"\nFixed baseline found: ef={ef}"
            )

            return result, all_results

    print(
        "\nWARNING:"
        " No Fixed efSearch can satisfy "
        f"Coverage >= {TARGET_COVERAGE:.2%}"
    )

    return None, all_results


# ============================================================
# Oracle
#
# 对每个 Query：
# 找到达到 Target Recall 所需的最小 ef
# ============================================================

def calculate_oracle(
    index,
    queries,
    ground_truth,
    k
):

    print("\n" + "=" * 70)
    print("Calculating Oracle")
    print("=" * 70)

    num_queries = len(queries)

    optimal_ef = np.full(
        num_queries,
        np.nan,
        dtype=np.float32
    )

    # 逐 ef 搜索
    for ef in ORACLE_EF_CANDIDATES:

        print(f"Oracle testing ef={ef}")

        index.set_ef(ef)

        labels, _ = index.knn_query(
            queries,
            k=k
        )

        recalls = calculate_query_recall(
            labels,
            ground_truth,
            k
        )

        # 尚未找到最优 ef 的 Query
        unfinished = np.isnan(optimal_ef)

        # 当前已经满足目标 Recall
        satisfied = recalls >= TARGET_RECALL

        update = unfinished & satisfied

        optimal_ef[update] = ef

        print(
            f"Currently solved: "
            f"{np.sum(~np.isnan(optimal_ef))}"
            f"/{num_queries}"
        )

    # 能找到 Oracle 的 Query
    valid = ~np.isnan(optimal_ef)

    oracle_coverage = np.mean(valid)

    if np.sum(valid) == 0:

        print("No query reaches target Recall.")

        return None

    oracle_average_ef = np.mean(
        optimal_ef[valid]
    )

    oracle_median_ef = np.median(
        optimal_ef[valid]
    )

    print("\nOracle result:")
    print(
        f"Coverage = {oracle_coverage:.4%}"
    )

    print(
        f"Average optimal ef = "
        f"{oracle_average_ef:.2f}"
    )

    print(
        f"Median optimal ef = "
        f"{oracle_median_ef:.2f}"
    )

    return {
        "optimal_ef": optimal_ef,
        "coverage": oracle_coverage,
        "average_ef": oracle_average_ef,
        "median_ef": oracle_median_ef,
        "valid": valid,
    }


# ============================================================
# 运行 Oracle 的实际搜索
#
# 注意：
# Oracle 是理论模型。
# 这里为了得到 Latency/QPS，
# 按照每个 Query 的 optimal ef 真正执行一次搜索。
# ============================================================

def run_oracle_search(
    index,
    queries,
    ground_truth,
    oracle,
    k
):

    optimal_ef = oracle["optimal_ef"]

    valid = ~np.isnan(optimal_ef)

    valid_indices = np.where(valid)[0]

    total_time = 0.0

    all_recalls = []

    print("\nRunning Oracle search...")

    for idx in valid_indices:

        ef = int(optimal_ef[idx])

        index.set_ef(ef)

        start = time.perf_counter()

        labels, _ = index.knn_query(
            queries[idx:idx + 1],
            k=k
        )

        total_time += (
            time.perf_counter() - start
        )

        recall = calculate_query_recall(
            labels,
            ground_truth[idx:idx + 1],
            k
        )[0]

        all_recalls.append(recall)

    if len(valid_indices) == 0:
        return None

    average_recall = np.mean(
        all_recalls
    )

    average_ef = np.mean(
        optimal_ef[valid]
    )

    latency_ms = (
        total_time /
        len(valid_indices) *
        1000
    )

    qps = (
        len(valid_indices) /
        total_time
    )

    return {
        "average_recall": average_recall,
        "coverage": oracle["coverage"],
        "average_ef": average_ef,
        "latency_ms": latency_ms,
        "qps": qps,
    }


# ============================================================
# 生成最终 Fixed vs Oracle 表
# ============================================================

def make_comparison_table(
    dataset_name,
    fixed,
    oracle_result
):

    rows = []

    # Fixed
    if fixed is not None:

        rows.append({
            "Dataset": dataset_name,
            "Method": "Fixed",
            "TopK": TOPK,
            "Target Recall": TARGET_RECALL,
            "Coverage": fixed["coverage"],
            "Average Recall": fixed["average_recall"],
            "Average efSearch": fixed["ef"],
            "Latency (ms)": fixed["latency_ms"],
            "QPS": fixed["qps"],
        })

    # Oracle
    if oracle_result is not None:

        rows.append({
            "Dataset": dataset_name,
            "Method": "Oracle",
            "TopK": TOPK,
            "Target Recall": TARGET_RECALL,
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

def run_dataset(
    dataset_name,
    path
):

    base, queries, ground_truth = \
        load_dataset(path)

    space = get_space(path)

    index = build_index(
        base,
        space
    )

    # --------------------------------------------------------
    # Fixed
    # --------------------------------------------------------

    fixed, fixed_all = find_fixed_baseline(
        index,
        queries,
        ground_truth,
        TOPK
    )

    # --------------------------------------------------------
    # Oracle
    # --------------------------------------------------------

    oracle = calculate_oracle(
        index,
        queries,
        ground_truth,
        TOPK
    )

    if oracle is not None:

        oracle_result = run_oracle_search(
            index,
            queries,
            ground_truth,
            oracle,
            TOPK
        )

    else:

        oracle_result = None

    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    rows = make_comparison_table(
        dataset_name,
        fixed,
        oracle_result
    )

    return rows, fixed_all, oracle


# ============================================================
# Main
# ============================================================

def main():

    os.makedirs(
        RESULT_DIR,
        exist_ok=True
    )

    all_rows = []

    print("\n")
    print("=" * 80)
    print("FIXED vs ORACLE EXPERIMENT")
    print("=" * 80)

    print(
        f"TopK           = {TOPK}"
    )

    print(
        f"Target Recall  = {TARGET_RECALL}"
    )

    print(
        f"Target Coverage = {TARGET_COVERAGE}"
    )

    # ========================================================
    # 遍历数据集
    # ========================================================

    for dataset_name, path in DATASETS.items():

        if not os.path.exists(path):

            print(
                f"\nWARNING: "
                f"{path} does not exist."
            )

            continue

        rows, _, _ = run_dataset(
            dataset_name,
            path
        )

        all_rows.extend(rows)

    # ========================================================
    # 保存最终表格
    # ========================================================

    if len(all_rows) == 0:

        print(
            "\nNo valid experiment results."
        )

        return

    df = pd.DataFrame(
        all_rows
    )

    output_path = os.path.join(
        RESULT_DIR,
        "fixed_vs_oracle.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    # ========================================================
    # 打印最终结果
    # ========================================================

    print("\n")
    print("=" * 100)
    print("FINAL FIXED vs ORACLE COMPARISON")
    print("=" * 100)

    print(
        df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}"
        )
    )

    print("\n")
    print(
        f"Results saved to: "
        f"{output_path}"
    )


if __name__ == "__main__":
    main()
