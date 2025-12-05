import os
import sys
import glob
import json
import re
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    data = []
    if not os.path.exists(file_path):
        logger.warning(f"File not found: {file_path}")
        return data
        
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return data

def extract_series_and_baselines(data):
    series = []
    baseline_no_conf = None
    
    for entry in data:
        agg_type = entry.get('aggregation_type')
        acc = entry.get('accuracy', 0.0)
        
        if agg_type == 'baseline_to_baseline_aggregation_without_confidence':
            baseline_no_conf = acc
        elif agg_type == 'baseline_to_aggllm_aggregation':
            ckpt = entry.get('checkpoint')
            if ckpt is not None:
                series.append((ckpt, acc))
                
    series.sort(key=lambda x: x[0])
    return series, baseline_no_conf

def plot_single_aggregation(benchmark_name, aggllm_dir, output_dir, base_dir):
    file_name = f"{benchmark_name}_accuracy_all_checkpoints.jsonl"
    path = os.path.join(aggllm_dir, file_name)
    
    data = load_data(path)
    
    if not data:
        logger.warning(f"No data found for {benchmark_name}")
        return

    # Extract data
    series, baseline_no_conf = extract_series_and_baselines(data)

    # Load extra metrics
    metrics_file = os.path.join(base_dir, f"{benchmark_name}_metrics.json")
    metrics_data = {}
    if os.path.exists(metrics_file):
        try:
            with open(metrics_file, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            logger.info(f"Loaded metrics from {metrics_file}")
        except Exception as e:
            logger.warning(f"Failed to load metrics file {metrics_file}: {e}")
    else:
        logger.warning(f"Metrics file not found: {metrics_file}")

    # Plotting
    plt.figure(figsize=(14, 9))
    
    # Plot AggLLM series
    if series:
        x, y = zip(*series)
        plt.plot(x, y, marker='o', label='AggLLM', color='blue', linewidth=2)
        
    # Plot Prompt Agg baseline (without confidence)
    if baseline_no_conf is not None:
        plt.axhline(y=baseline_no_conf, color='green', linestyle='--', 
                   linewidth=2, label='Prompt Agg')

    # Plot Extra Metrics (Pass@1 and Majority@8 only, with dashed lines)
    extra_metrics_config = {
        "pass_at_1": ("Pass@1", "purple", "--"),
        "majority_voting_set_8": ("Majority@8", "navy", "--")
    }

    for key, (label, color, style) in extra_metrics_config.items():
        val = metrics_data.get("baseline", {}).get(key)
        if val is not None:
            plt.axhline(y=val, color=color, linestyle=style, 
                       linewidth=2, label=label, alpha=0.9)
        
    plt.title(f'{benchmark_name} Aggregation Results')
    plt.xlabel('Checkpoint')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{benchmark_name}_aggregation.png')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved aggregation plot to {output_path}")

def main():
    base_dir = "/root/projects/Conf_Agg/output_s/outputs/comprehensive_results/Qwen_Qwen3-1.7B_think_True_32768"
    aggllm_dir_name = "4000_32_baseline"  # AggLLM 결과 폴더
    
    aggllm_dir = os.path.join(base_dir, aggllm_dir_name)
    
    output_dir = os.path.join(base_dir, "aggregation_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all benchmarks in aggllm_dir
    files = glob.glob(os.path.join(aggllm_dir, "*_accuracy_all_checkpoints.jsonl"))
    benchmarks = set()
    for f in files:
        filename = os.path.basename(f)
        bench_name = filename.replace('_accuracy_all_checkpoints.jsonl', '')
        benchmarks.add(bench_name)
    
    logger.info(f"Found benchmarks: {benchmarks}")
    
    for bench in benchmarks:
        plot_single_aggregation(bench, aggllm_dir, output_dir, base_dir)

if __name__ == "__main__":
    main()

