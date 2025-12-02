#!/usr/bin/env python3
"""
Aggregation accuracy + metrics 시각화 스크립트.
- 주 accuracy 폴더: Prompt Agg (w/ conf, w/o conf) + 지정 라벨의 baseline→AggLLM
- 비교 폴더: label=path 형태로 전달하면 각 폴더의 baseline_to_aggllm을 추가 라인으로 표시
- metrics json: pass@1/4/8은 수평선, majority/bottom 10%는 set-4/8을 잇는 라인으로 표현
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt


TARGET_TYPES = [
    ("baseline_to_baseline_aggregation", "Prompt Agg w/ conf"),
    ("baseline_to_baseline_aggregation_without_confidence", "Prompt Agg w/o conf"),
]

PASS_KEYS = [
    ("pass_at_1", "pass@1", "#d62728", "--"),
    ("pass_at_4", "pass@4", "#2ca02c", ":"),
    ("pass_at_8", "pass@8", "#9467bd", "-."),
]
MAJORITY_KEYS = ("majority_voting_set_4", "majority_voting_set_8", "Majority Voting", "#1f77b4")
BOTTOM_KEYS = (
    "confidence_bottom_10_percent_confidence_set_4",
    "confidence_bottom_10_percent_confidence_set_8",
    "Bottom 10% Confidence",
    "#ff7f0e",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot aggregation accuracies with additional metrics."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--accuracy_file", help="주 accuracy jsonl (단일 실행)")
    group.add_argument(
        "--accuracy_dir",
        help="주 accuracy 폴더 (여기 있는 *_accuracy_checkpoint_2400.jsonl 모두 처리)",
    )

    parser.add_argument("--metrics_file", help="단일 실행용 metrics json")
    parser.add_argument("--metrics_dir", help="디렉터리 모드에서 metrics json 위치")
    parser.add_argument("--output_file", help="단일 실행 결과 경로")
    parser.add_argument("--output_dir", help="디렉터리 모드 출력 폴더 (기본: accuracy와 동일)")
    parser.add_argument("--title", help="단일 실행 그래프 제목")

    parser.add_argument(
        "--primary_dir_label",
        help="주 폴더 baseline→AggLLM 라벨 (기본: 폴더 이름)",
    )
    parser.add_argument(
        "--compare_dirs",
        nargs="*",
        help="추가 비교 폴더 label=path 형식 (예: 'baseline->AggLLM w/o conf=/path/...')",
    )
    return parser.parse_args()


def load_accuracy_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def prepare_series(rows: List[Dict]) -> Dict[str, Dict[int, float]]:
    series: Dict[str, Dict[int, float]] = {
        "baseline_to_baseline_aggregation": {},
        "baseline_to_baseline_aggregation_without_confidence": {},
        "baseline_to_aggllm_aggregation": {},
    }
    for row in rows:
        agg_type = row.get("aggregation_type")
        if agg_type not in series:
            continue
        group_size = int(row.get("group_size"))
        series[agg_type][group_size] = row.get("accuracy")
    return series


def load_metrics(path: Path) -> Dict[str, float]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    baseline = data.get("baseline", {})
    metrics = {}
    for key, *_ in PASS_KEYS:
        metrics[key] = baseline.get(key)
    for key in (MAJORITY_KEYS[0], MAJORITY_KEYS[1], BOTTOM_KEYS[0], BOTTOM_KEYS[1]):
        metrics[key] = baseline.get(key)
    return metrics


def compute_ylim(values: Sequence[Optional[float]]) -> Optional[tuple]:
    vals = [v for v in values if isinstance(v, (int, float))]
    if not vals:
        return None
    min_v = min(vals)
    max_v = max(vals)
    if min_v == max_v:
        min_v -= 0.05
        max_v += 0.05
    margin = (max_v - min_v) * 0.2 + 1e-4
    return (max(0.0, min_v - margin), min(1.05, max_v + margin))


def sanitize_dataset_name(name: str) -> str:
    return name.replace("/", "_")


def dataset_name_from_accuracy(rows: List[Dict]) -> str:
    if not rows:
        return "Unknown"
    row = rows[0]
    return row.get("dataset_path") or row.get("dataset_name") or "Unknown"


def parse_labelled_dirs(values: Optional[List[str]]) -> List[Tuple[str, Path]]:
    result: List[Tuple[str, Path]] = []
    if not values:
        return result
    for item in values:
        if "=" not in item:
            raise ValueError(f"--compare_dirs 항목은 label=path 형식이어야 합니다: {item}")
        label, path_str = item.split("=", 1)
        result.append((label.strip(), Path(path_str.strip()).resolve()))
    return result


def find_matching_file(target_name: str, directory: Path) -> Optional[Path]:
    direct = directory / target_name
    if direct.exists():
        return direct
    candidates = list(directory.glob(f"*{Path(target_name).stem}*.jsonl"))
    return candidates[0] if candidates else None


def plot_combined(
    group_sizes: List[int],
    prompt_series: Dict[str, Dict[int, float]],
    agg_series: List[Tuple[str, Dict[int, float], Dict]],
    metrics: Dict[str, float],
    title: str,
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    line_defs = [
        ("Prompt Agg w/ conf", prompt_series.get("baseline_to_baseline_aggregation", {}), {}),
        (
            "Prompt Agg w/o conf",
            prompt_series.get("baseline_to_baseline_aggregation_without_confidence", {}),
            {},
        ),
    ]
    line_defs.extend(agg_series)

    for label, data, style in line_defs:
        if not data:
            continue
        y_values = [data.get(gs) for gs in group_sizes]
        ax.plot(
            group_sizes,
            y_values,
            marker=style.get("marker", "o"),
            linewidth=style.get("linewidth", 2.0),
            markersize=style.get("markersize", 7),
            linestyle=style.get("linestyle", "-"),
            color=style.get("color"),
            label=label,
        )

    # Determine x-span
    x_min = min(group_sizes + [4])
    x_max = max(group_sizes + [8])

    # Pass@k horizontal lines
    for key, label, color, linestyle in PASS_KEYS:
        val = metrics.get(key)
        if val is None:
            continue
        ax.hlines(
            y=val,
            xmin=x_min,
            xmax=x_max,
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
            label=label,
        )

    def plot_pair(keys, label, color):
        v4 = metrics.get(keys[0])
        v8 = metrics.get(keys[1])
        if v4 is None or v8 is None:
            return
        ax.plot(
            [4, 8],
            [v4, v8],
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )

    plot_pair(MAJORITY_KEYS[:2], MAJORITY_KEYS[2], MAJORITY_KEYS[3])
    plot_pair(BOTTOM_KEYS[:2], BOTTOM_KEYS[2], BOTTOM_KEYS[3])

    ax.set_xlabel("Group Size / Set Size")
    ax.set_ylabel("Score")
    ax.set_title(title)
    y_values_all = [
        v
        for _, data, _ in line_defs
        for v in (data or {}).values()
        if isinstance(v, (int, float))
    ]
    metric_vals = [
        metrics.get(key)
        for key in [pk[0] for pk in PASS_KEYS]
        + [MAJORITY_KEYS[0], MAJORITY_KEYS[1], BOTTOM_KEYS[0], BOTTOM_KEYS[1]]
    ]
    y_values_all.extend(v for v in metric_vals if isinstance(v, (int, float)))
    ylim = compute_ylim(y_values_all)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlim(x_min - 0.5, x_max + 0.5)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_compare_series(
    compare_dirs: List[Tuple[str, Path]],
    dataset_filename: str,
) -> List[Tuple[str, Dict[int, float], Dict]]:
    series_list: List[Tuple[str, Dict[int, float], Dict]] = []
    for label, dir_path in compare_dirs:
        match = find_matching_file(dataset_filename, dir_path)
        if not match:
            print(f"⚠️  {label}: {dataset_filename} 미존재, 건너뜀")
            continue
        comp_rows = load_accuracy_rows(match)
        comp_series = prepare_series(comp_rows)
        series_list.append(
            (
                label,
                comp_series.get("baseline_to_aggllm_aggregation", {}),
                {},
            )
        )
    return series_list


def run_single(
    accuracy_file: Path,
    metrics_file: Path,
    output_file: Path,
    title: Optional[str],
    primary_label: Optional[str],
    compare_dirs: List[Tuple[str, Path]],
) -> None:
    rows = load_accuracy_rows(accuracy_file)
    if not rows:
        raise ValueError(f"No data in {accuracy_file}")

    dataset_name = dataset_name_from_accuracy(rows)
    if not metrics_file.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    prompt_series = prepare_series(rows)
    group_sizes = sorted(
        {gs for data in prompt_series.values() for gs in data.keys()}
    )
    metrics = load_metrics(metrics_file)

    compare_series = [
        (
            primary_label or "AggLLM",
            prompt_series.get("baseline_to_aggllm_aggregation", {}),
            {"linewidth": 2.5, "color": "#d62728"},
        )
    ]
    compare_series.extend(build_compare_series(compare_dirs, accuracy_file.name))

    output_file.parent.mkdir(parents=True, exist_ok=True)
    plot_combined(
        group_sizes,
        prompt_series,
        compare_series,
        metrics,
        title or f"{dataset_name} Aggregation",
        output_file,
    )
    print(f"✅ 저장 완료: {output_file}")


def run_batch(
    accuracy_dir: Path,
    metrics_dir: Path,
    output_dir: Optional[Path],
    primary_label: Optional[str],
    compare_dirs: List[Tuple[str, Path]],
) -> None:
    accuracy_files = sorted(accuracy_dir.glob("*_accuracy_checkpoint_2400.jsonl"))
    if not accuracy_files:
        raise FileNotFoundError(f"No *_accuracy_checkpoint_2400.jsonl under {accuracy_dir}")

    for acc_file in accuracy_files:
        rows = load_accuracy_rows(acc_file)
        if not rows:
            continue
        dataset_name = dataset_name_from_accuracy(rows)
        metrics_filename = f"{sanitize_dataset_name(dataset_name)}_metrics.json"
        metrics_file = metrics_dir / metrics_filename

        if not metrics_file.exists():
            print(f"⚠️  metrics 파일을 찾을 수 없어 건너뜀: {metrics_file}")
            continue

        prompt_series = prepare_series(rows)
        group_sizes = sorted(
            {gs for data in prompt_series.values() for gs in data.keys()}
        )
        metrics = load_metrics(metrics_file)

        compare_series = [
            (
                primary_label or "AggLLM",
                prompt_series.get("baseline_to_aggllm_aggregation", {}),
                {"linewidth": 2.5, "color": "#d62728"},
            )
        ]
        compare_series.extend(build_compare_series(compare_dirs, acc_file.name))

        out_dir = output_dir or acc_file.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        output_file = out_dir / f"{sanitize_dataset_name(dataset_name)}_aggregation_summary.png"
        plot_combined(
            group_sizes,
            prompt_series,
            compare_series,
            metrics,
            f"{dataset_name} Aggregation",
            output_file,
        )
        print(f"✅ 저장 완료: {output_file}")


def main() -> None:
    args = parse_args()

    if args.accuracy_file:
        if not args.metrics_file or not args.output_file:
            raise ValueError("--metrics_file 및 --output_file 필요")
        run_single(
            Path(args.accuracy_file).resolve(),
            Path(args.metrics_file).resolve(),
            Path(args.output_file).resolve(),
            args.title,
            args.primary_dir_label,
            parse_labelled_dirs(args.compare_dirs),
        )
    else:
        if not args.metrics_dir:
            raise ValueError("--metrics_dir 필요")
        run_batch(
            Path(args.accuracy_dir).resolve(),
            Path(args.metrics_dir).resolve(),
            Path(args.output_dir).resolve() if args.output_dir else None,
            args.primary_dir_label,
            parse_labelled_dirs(args.compare_dirs),
        )


if __name__ == "__main__":
    main()

