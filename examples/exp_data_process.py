import re
from typing import Dict, List, Tuple, Optional
from pathlib import Path


CLIENT_PATTERN = re.compile(r'\[(?:[^\]]*?\s)?(Client\d+)\]')
ROUND_PATTERN = re.compile(r'\[(?:[^\]]*?\s)?(Client\d+)\]:\s*(\d+)\s+\d+\s+[YN]\b')
RATIO_PATTERN = re.compile(r'compression ratio:\s*([0-9]*\.?[0-9]+)')


def parse_log_file(file_path: Path) -> Dict[int, Dict[str, float]]:
    data: Dict[int, Dict[str, float]] = {}
    current_client: Optional[str] = None
    client_round: Dict[str, int] = {}

    with file_path.open("r") as handle:
        for line in handle:
            client_match = CLIENT_PATTERN.search(line)
            if client_match:
                current_client = client_match.group(1)

            round_match = ROUND_PATTERN.search(line)
            if round_match:
                client = round_match.group(1)
                round_id = int(round_match.group(2))
                client_round[client] = round_id
                data.setdefault(round_id, {})

            if "compression ratio:" in line:
                ratio_match = RATIO_PATTERN.search(line)
                if ratio_match and current_client:
                    round_id = client_round.get(current_client)
                    if round_id is not None:
                        ratio = float(ratio_match.group(1))
                        data.setdefault(round_id, {})[current_client] = ratio
    return data


def calculate_max_improvement(
    baseline: Dict[int, Dict[str, float]],
    optimized: Dict[int, Dict[str, float]]
) -> List[Tuple[int, str, float, float, float]]:
    results: List[Tuple[int, str, float, float, float]] = []
    for round_id in sorted(set(baseline) & set(optimized)):
        if round_id == 0:
            continue
        best_pct = float("-inf")
        best_entry: Optional[Tuple[int, str, float, float, float]] = None
        for client in set(baseline[round_id]) & set(optimized[round_id]):
            base_ratio = baseline[round_id][client]
            opt_ratio = optimized[round_id][client]
            if base_ratio <= 0:
                continue
            pct_gain = (opt_ratio - base_ratio) / base_ratio * 100.0
            if pct_gain > best_pct:
                best_pct = pct_gain
                best_entry = (round_id, client, base_ratio, opt_ratio, pct_gain)
        if best_entry:
            results.append(best_entry)
    return results


def summarize(results: List[Tuple[int, str, float, float, float]]) -> str:
    lines = ["=" * 72, "Compression Ratio Percentage Gain (per round)", "=" * 72, ""]
    if not results:
        lines.append("No overlapping rounds/clients with valid ratios were found.")
        lines.append("=" * 72)
        return "\n".join(lines)

    lines.append(f"{'Round':<8}{'Client':<12}{'Baseline':<14}{'Optimized':<14}{'Δ %':<12}")
    lines.append("-" * 72)

    total_pct = 0.0
    best_overall = max(results, key=lambda item: item[4])

    for round_id, client, base, opt, pct in results:
        lines.append(f"{round_id:<8}{client:<12}{base:<14.4f}{opt:<14.4f}{pct:<12.2f}")
        total_pct += pct

    lines.append("-" * 72)
    lines.append(f"Average Δ% per round: {total_pct / len(results):.2f}")
    lines.append(f"Best overall: Round {best_overall[0]}, {best_overall[1]} (Δ {best_overall[4]:.2f}%)")
    lines.append("=" * 72)
    return "\n".join(lines)


def analyze_pair(baseline_file: Path, optimized_file: Path, output_dir: Path) -> None:
    baseline = parse_log_file(baseline_file)
    optimized = parse_log_file(optimized_file)
    results = calculate_max_improvement(baseline, optimized)

    report = summarize(results)

    error_token = baseline_file.stem.split("rel_")[-1] if "rel_" in baseline_file.stem else baseline_file.stem
    output_path = output_dir / f"compression_ratio_analysis_rel_{error_token}.txt"
    output_path.write_text(report)

    if results:
        round_id, client, base, opt, pct = max(results, key=lambda item: item[4])
        print(
            f"[{error_token}] Round {round_id}, {client}: "
            f"baseline CR={base:.4f}, optimized CR={opt:.4f}, Δ={pct:.2f}%"
        )
    else:
        print(f"[{error_token}] No overlapping rounds/clients found.")


def main():
    root = Path("/eagle/lc-mpi/ZhijingYe/APPFL/examples/exp_output/Model_Wise/InceptionV3-Caltech101")
    if not root.exists():
        print("Log directory not found.")
        return

    output_dir = root / "analysis"
    output_dir.mkdir(exist_ok=True)

    pairs_found = False
    for baseline_file in sorted(root.glob("SZ3_m_w_rel_*.txt")):
        candidate_name = baseline_file.name.replace("SZ3", "TP", 1)
        optimized_file = root / candidate_name
        if optimized_file.exists():
            pairs_found = True
            analyze_pair(baseline_file, optimized_file, output_dir)

    if not pairs_found:
        print("No SZ3/TP log pairs detected.")


if __name__ == "__main__":
    main()