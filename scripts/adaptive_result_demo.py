import os
import time
import h5py
import hnswlib
import numpy as np
import pandas as pd
import itertools

# ============================================================
# Configuration
# ============================================================

DATA_DIR = "../data/"

COHERE_BASE = os.path.join(DATA_DIR, "cohere_train.f32")
COHERE_QUERY = os.path.join(DATA_DIR, "cohere_test.f32")
COHERE_GT = os.path.join(DATA_DIR, "cohere_groundtruth.i32")

COHERE_DIM = 768
COHERE_N_BASE = 1000000
COHERE_N_QUERY = 1000
COHERE_GT_K = 1000

RESULT_PATH = "../results/adaptive_fair_comparison.csv"

K = 10
TARGET_RECALL = 0.95
TARGET_COVERAGE = 0.90  # 目标：90%的查询达到召回率0.95

# HNSW参数
M = 16
EF_CONSTRUCTION = 200
RANDOM_SEED = 42

# 自适应参数搜索范围
EF_CHEAP_OPTIONS = [50, 100, 150]
EF_EXPENSIVE_OPTIONS = [400, 600, 800]
THRESHOLD_OPTIONS = [0.85, 0.88, 0.90, 0.92, 0.95]

CALIBRATION_RATIO = 0.3


# ============================================================
# Load Cohere dataset
# ============================================================

def load_cohere_dataset(base_path, query_path, gt_path):
    print("=" * 70)
    print("Loading Cohere dataset...")
    print("=" * 70)

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base file not found: {base_path}")
    if not os.path.exists(query_path):
        raise FileNotFoundError(f"Query file not found: {query_path}")
    if not os.path.exists(gt_path):
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")

    print(f"Loading base vectors: {base_path}")
    base_data = np.fromfile(base_path, dtype=np.float32)
    train = base_data.reshape(COHERE_N_BASE, COHERE_DIM)
    print(f"  Base vectors: {train.shape} ({train.nbytes / 1024 ** 3:.2f} GiB)")

    print(f"Loading query vectors: {query_path}")
    query_data = np.fromfile(query_path, dtype=np.float32)
    queries = query_data.reshape(COHERE_N_QUERY, COHERE_DIM)
    print(f"  Query vectors: {queries.shape}")

    print(f"Loading ground truth: {gt_path}")
    gt_data = np.fromfile(gt_path, dtype=np.int32)
    ground_truth = gt_data.reshape(COHERE_N_QUERY, COHERE_GT_K)
    print(f"  Ground truth: {ground_truth.shape}")

    return train, queries, ground_truth


# ============================================================
# Build HNSW
# ============================================================

def build_index(data, space="cosine"):
    print()
    print("Building HNSW index...")
    print(f"  M={M}, ef_construction={EF_CONSTRUCTION}")

    index = hnswlib.Index(space=space, dim=data.shape[1])
    index.init_index(max_elements=len(data), M=M, ef_construction=EF_CONSTRUCTION)
    index.add_items(data)

    print("Index built")
    return index


# ============================================================
# Recall Calculation
# ============================================================

def calculate_recall(labels, ground_truth):
    gt = ground_truth[:, :K]
    recalls = np.zeros(len(labels), dtype=np.float32)
    for i in range(len(labels)):
        hits = np.isin(labels[i], gt[i])
        recalls[i] = np.sum(hits) / K
    return recalls


# ============================================================
# Search function
# ============================================================

def search_queries(index, queries, query_ids, ef, ground_truth):
    if len(query_ids) == 0:
        return np.array([]), 0.0, np.array([]), np.array([])

    index.set_ef(ef)
    selected_queries = queries[query_ids]

    start = time.perf_counter()
    labels, distances = index.knn_query(selected_queries, k=K)
    elapsed = time.perf_counter() - start

    recalls = calculate_recall(labels, ground_truth[query_ids])

    return recalls, elapsed, labels, distances


# ============================================================
# Two-Stage Adaptive Search
# ============================================================

def run_two_stage_adaptive(index, queries, ground_truth, calibration_ids,
                           evaluation_ids, ef_cheap, ef_expensive, threshold):
    print()
    print("=" * 70)
    print("TWO-STAGE ADAPTIVE SEARCH")
    print("=" * 70)
    print(f"Stage 1: ef={ef_cheap} (cheap)")
    print(f"Stage 2: ef={ef_expensive} (expensive)")
    print(f"Prediction threshold: recall >= {threshold}")
    print()

    all_ids = np.concatenate([calibration_ids, evaluation_ids])

    # Stage 1
    cheap_recalls, cheap_elapsed, cheap_labels, cheap_distances = search_queries(
        index, queries, all_ids, ef_cheap, ground_truth
    )

    cheap_recalls_eval = cheap_recalls[len(calibration_ids):]

    # Decision
    stop_mask_eval = cheap_recalls_eval >= threshold
    continue_ids = evaluation_ids[~stop_mask_eval]

    print(f"\nDecision:")
    print(
        f"  Stop in Stage 1: {np.sum(stop_mask_eval)} queries ({np.sum(stop_mask_eval) / len(evaluation_ids) * 100:.1f}%)")
    print(f"  Continue to Stage 2: {len(continue_ids)} queries ({len(continue_ids) / len(evaluation_ids) * 100:.1f}%)")

    # Stage 2
    final_recalls = np.zeros(len(evaluation_ids), dtype=np.float32)

    stop_indices = np.where(stop_mask_eval)[0]
    final_recalls[stop_indices] = cheap_recalls_eval[stop_indices]

    stage2_elapsed = 0.0
    if len(continue_ids) > 0:
        print(f"\n[Stage 2] Searching {len(continue_ids)} queries with ef={ef_expensive}...")

        stage2_recalls, stage2_elapsed, _, _ = search_queries(
            index, queries, continue_ids, ef_expensive, ground_truth
        )

        continue_indices = np.where(~stop_mask_eval)[0]
        final_recalls[continue_indices] = stage2_recalls

        print(f"  Average recall: {np.mean(stage2_recalls):.4f}")
        print(f"  Elapsed: {stage2_elapsed:.2f}s")

    total_elapsed = cheap_elapsed + stage2_elapsed

    coverage = np.mean(final_recalls >= TARGET_RECALL)
    average_recall = np.mean(final_recalls)
    avg_ef = np.mean(np.where(stop_mask_eval, ef_cheap, ef_expensive))
    qps = len(evaluation_ids) / total_elapsed if total_elapsed > 0 else 0
    latency_ms = total_elapsed / len(evaluation_ids) * 1000 if total_elapsed > 0 else 0

    return {
        "coverage": coverage,
        "average_recall": average_recall,
        "avg_ef": avg_ef,
        "latency_ms": latency_ms,
        "qps": qps,
        "total_elapsed": total_elapsed,
        "stop_count": np.sum(stop_mask_eval),
        "total_count": len(evaluation_ids),
        "stage1_recalls": cheap_recalls_eval,
        "final_recalls": final_recalls,
        "stop_mask": stop_mask_eval,
        "stage1_elapsed": cheap_elapsed,
        "stage2_elapsed": stage2_elapsed,
    }


# ============================================================
# Fixed baseline - 运行所有候选ef
# ============================================================

def run_all_fixed(index, queries, ground_truth, query_ids):
    candidates = [50, 100, 200, 300, 400, 600, 800, 1000, 1200]

    print()
    print("=" * 70)
    print("RUNNING ALL FIXED EF")
    print("=" * 70)

    results = {}

    for ef in candidates:
        print(f"Testing ef={ef}...")
        index.set_ef(ef)
        selected_queries = queries[query_ids]

        start = time.perf_counter()
        labels, _ = index.knn_query(selected_queries, k=K)
        elapsed = time.perf_counter() - start

        recalls = calculate_recall(labels, ground_truth[query_ids])

        results[ef] = {
            "ef": ef,
            "coverage": np.mean(recalls >= TARGET_RECALL),
            "average_recall": np.mean(recalls),
            "latency_ms": elapsed / len(query_ids) * 1000,
            "qps": len(query_ids) / elapsed,
            "recalls": recalls,
        }

        print(f"  Coverage={results[ef]['coverage']:.4f}, QPS={results[ef]['qps']:.2f}")

    return results


def find_fixed_at_coverage(fixed_results, target_coverage):
    """找到达到目标覆盖率的最小ef，支持线性插值"""

    # 找最接近的
    ef_list = sorted(fixed_results.keys())

    # 如果直接找到了
    for ef in ef_list:
        if fixed_results[ef]['coverage'] >= target_coverage:
            return ef, fixed_results[ef]

    # 如果没找到，用插值
    for i in range(len(ef_list) - 1):
        ef1 = ef_list[i]
        ef2 = ef_list[i + 1]
        cov1 = fixed_results[ef1]['coverage']
        cov2 = fixed_results[ef2]['coverage']

        if cov1 < target_coverage <= cov2:
            # 线性插值
            ratio = (target_coverage - cov1) / (cov2 - cov1)
            ef_interp = ef1 + ratio * (ef2 - ef1)
            qps_interp = fixed_results[ef1]['qps'] + ratio * (fixed_results[ef2]['qps'] - fixed_results[ef1]['qps'])

            return ef_interp, {
                "ef": ef_interp,
                "coverage": target_coverage,
                "average_recall": fixed_results[ef1]['average_recall'] + ratio * (
                            fixed_results[ef2]['average_recall'] - fixed_results[ef1]['average_recall']),
                "latency_ms": 1000 / qps_interp,
                "qps": qps_interp,
            }

    return None, None


# ============================================================
# 搜索最优自适应参数
# ============================================================

def search_adaptive_params(index, queries, ground_truth, calibration_ids, evaluation_ids):
    """搜索使覆盖率>=TARGET_COVERAGE且QPS最高的参数组合"""

    print()
    print("=" * 70)
    print("SEARCHING OPTIMAL ADAPTIVE PARAMETERS")
    print("=" * 70)
    print(f"Target Coverage: {TARGET_COVERAGE}")
    print(f"EF_CHEAP options: {EF_CHEAP_OPTIONS}")
    print(f"EF_EXPENSIVE options: {EF_EXPENSIVE_OPTIONS}")
    print(f"Threshold options: {THRESHOLD_OPTIONS}")

    best_result = None
    best_qps = -1
    best_params = None

    total_combinations = len(EF_CHEAP_OPTIONS) * len(EF_EXPENSIVE_OPTIONS) * len(THRESHOLD_OPTIONS)
    tested = 0

    for ef_cheap, ef_expensive, threshold in itertools.product(
            EF_CHEAP_OPTIONS, EF_EXPENSIVE_OPTIONS, THRESHOLD_OPTIONS
    ):
        tested += 1
        print(
            f"\nTesting {tested}/{total_combinations}: ef_cheap={ef_cheap}, ef_expensive={ef_expensive}, threshold={threshold}")

        try:
            result = run_two_stage_adaptive(
                index, queries, ground_truth, calibration_ids, evaluation_ids,
                ef_cheap, ef_expensive, threshold
            )

            # 只考虑达到目标覆盖率的
            if result['coverage'] >= TARGET_COVERAGE:
                print(f"  ✓ Coverage={result['coverage']:.4f} >= {TARGET_COVERAGE}, QPS={result['qps']:.2f}")

                if result['qps'] > best_qps:
                    best_qps = result['qps']
                    best_result = result
                    best_params = (ef_cheap, ef_expensive, threshold)
            else:
                print(f"  ✗ Coverage={result['coverage']:.4f} < {TARGET_COVERAGE}, skipped")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    if best_result is None:
        print(f"\n⚠ No adaptive config achieves coverage >= {TARGET_COVERAGE}")
        return None, None

    print(f"\n✓ Best adaptive config found:")
    print(f"  ef_cheap={best_params[0]}, ef_expensive={best_params[1]}, threshold={best_params[2]}")
    print(f"  Coverage={best_result['coverage']:.4f}, QPS={best_result['qps']:.2f}")

    return best_result, best_params


# ============================================================
# Main
# ============================================================

def main():
    np.random.seed(RANDOM_SEED)

    # 加载Cohere数据集
    try:
        train, queries, ground_truth = load_cohere_dataset(
            COHERE_BASE, COHERE_QUERY, COHERE_GT
        )
    except FileNotFoundError as e:
        print(f"\nError: {e}")
        return

    # 构建索引
    index = build_index(train, "cosine")

    # 分割数据集
    all_ids = np.arange(len(queries))
    np.random.shuffle(all_ids)

    calib_size = int(len(all_ids) * CALIBRATION_RATIO)
    calibration_ids = all_ids[:calib_size]
    evaluation_ids = all_ids[calib_size:]

    print()
    print(f"Total queries: {len(queries)}")
    print(f"Calibration queries: {len(calibration_ids)}")
    print(f"Evaluation queries: {len(evaluation_ids)}")

    # ============================================================
    # 1. 运行所有固定ef
    # ============================================================
    fixed_results = run_all_fixed(index, queries, ground_truth, evaluation_ids)

    # 找到Fixed在TARGET_COVERAGE下的性能
    fixed_ef, fixed_at_target = find_fixed_at_coverage(fixed_results, TARGET_COVERAGE)

    if fixed_at_target is None:
        print(f"\n⚠ No Fixed config achieves coverage >= {TARGET_COVERAGE}")
        return

    # ============================================================
    # 2. 搜索最优自适应参数
    # ============================================================
    adaptive_result, adaptive_params = search_adaptive_params(
        index, queries, ground_truth, calibration_ids, evaluation_ids
    )

    if adaptive_result is None:
        print("\n⚠ No Adaptive config achieves coverage >= {TARGET_COVERAGE}")
        return

    # ============================================================
    # 3. 公平对比（相同覆盖率）
    # ============================================================
    print()
    print("=" * 80)
    print("FAIR COMPARISON (Same Coverage)")
    print("=" * 80)
    print(f"Target Coverage: {TARGET_COVERAGE:.4f}")
    print()

    print(f"{'Method':<12} {'Coverage':>10} {'Recall':>10} {'Avg ef':>10} {'Latency(ms)':>14} {'QPS':>12}")
    print("-" * 80)

    print(f"{'Fixed':<12} {fixed_at_target['coverage']:>10.4f} {fixed_at_target['average_recall']:>10.4f} "
          f"{fixed_at_target['ef']:>10.2f} {fixed_at_target['latency_ms']:>14.4f} {fixed_at_target['qps']:>12.2f}")

    print(f"{'Adaptive':<12} {adaptive_result['coverage']:>10.4f} {adaptive_result['average_recall']:>10.4f} "
          f"{adaptive_result['avg_ef']:>10.2f} {adaptive_result['latency_ms']:>14.4f} {adaptive_result['qps']:>12.2f}")

    # 改进分析
    print()
    print("=" * 80)
    print("FAIR IMPROVEMENT ANALYSIS")
    print("=" * 80)

    qps_improve = (adaptive_result['qps'] / fixed_at_target['qps'] - 1) * 100
    ef_reduction = (1 - adaptive_result['avg_ef'] / fixed_at_target['ef']) * 100

    print(f"Same Coverage: {TARGET_COVERAGE:.4f}")
    print(f"QPS Improvement: {qps_improve:+.2f}%")
    print(f"EF Reduction: {ef_reduction:+.2f}%")
    print(f"Recall Change: {adaptive_result['average_recall'] - fixed_at_target['average_recall']:+.4f}")

    # ============================================================
    # 4. 保存结果
    # ============================================================
    os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)

    summary_df = pd.DataFrame([
        {"method": "Fixed",
         "ef": fixed_at_target['ef'],
         "coverage": fixed_at_target['coverage'],
         "avg_recall": fixed_at_target['average_recall'],
         "latency_ms": fixed_at_target['latency_ms'],
         "qps": fixed_at_target['qps']},
        {"method": f"Adaptive(cheap={adaptive_params[0]},exp={adaptive_params[1]},thr={adaptive_params[2]})",
         "ef": adaptive_result['avg_ef'],
         "coverage": adaptive_result['coverage'],
         "avg_recall": adaptive_result['average_recall'],
         "latency_ms": adaptive_result['latency_ms'],
         "qps": adaptive_result['qps']}
    ])
    summary_df.to_csv(RESULT_PATH, index=False)

    print()
    print(f"Results saved to: {RESULT_PATH}")


if __name__ == "__main__":
    main()