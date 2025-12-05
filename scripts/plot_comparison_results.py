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
    baseline_conf = None
    baseline_no_conf = None
    
    for entry in data:
        agg_type = entry.get('aggregation_type')
        acc = entry.get('accuracy', 0.0)
        
        if agg_type == 'baseline_to_baseline_aggregation':
            baseline_conf = acc
        elif agg_type == 'baseline_to_baseline_aggregation_without_confidence':
            baseline_no_conf = acc
        elif agg_type == 'baseline_to_aggllm_aggregation':
            ckpt = entry.get('checkpoint')
            if ckpt is not None:
                series.append((ckpt, acc))
                
    series.sort(key=lambda x: x[0])
    return series, baseline_conf, baseline_no_conf

def plot_comparison(benchmark_name, dir1, dir2, output_dir, base_dir):
    file_name = f"{benchmark_name}_accuracy_all_checkpoints.jsonl"
    path1 = os.path.join(dir1, file_name)
    path2 = os.path.join(dir2, file_name)
    
    data1 = load_data(path1)
    data2 = load_data(path2)
    
    if not data1 and not data2:
        logger.warning(f"No data found for {benchmark_name}")
        return

    # Extract data
    # Dir1: baseline_kl0 -> w/o confidence
    series1, base_conf1, base_no_conf1 = extract_series_and_baselines(data1)
    
    # Dir2: trained_with_confidence_kl0 -> w confidence
    series2, base_conf2, base_no_conf2 = extract_series_and_baselines(data2)
    
    # Calculate average baselines
    avg_base_conf = None
    if base_conf1 is not None and base_conf2 is not None:
        avg_base_conf = (base_conf1 + base_conf2) / 2.0
    elif base_conf1 is not None:
        avg_base_conf = base_conf1
    elif base_conf2 is not None:
        avg_base_conf = base_conf2
        
    avg_base_no_conf = None
    if base_no_conf1 is not None and base_no_conf2 is not None:
        avg_base_no_conf = (base_no_conf1 + base_no_conf2) / 2.0
    elif base_no_conf1 is not None:
        avg_base_no_conf = base_no_conf1
    elif base_no_conf2 is not None:
        avg_base_no_conf = base_no_conf2

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
    
    # Plot Series 2 (w confidence) - now blue
    if series2:
        x2, y2 = zip(*series2)
        plt.plot(x2, y2, marker='s', label='AggLLM (w\o confidence)', color='orange')
        
    # Plot Series 1 (w/o confidence) - now orange
    if series1:
        x1, y1 = zip(*series1)
        plt.plot(x1, y1, marker='o', label='AggLLM (w confidence)', color='blue')
        
    # Plot Baselines
    if avg_base_conf is not None:
        plt.axhline(y=avg_base_conf, color='r', linestyle='--', label='Prompt Agg (with Conf)')
        
    if avg_base_no_conf is not None:
        plt.axhline(y=avg_base_no_conf, color='g', linestyle='--', label='Prompt Agg (w/o Conf)')

    # Plot Extra Metrics
    # Format: key: (Label, Color, Linestyle)
    extra_metrics_config = {
        "pass_at_1": ("Pass@1", "purple", "-"),
        # "pass_at_8": ("Pass@8", "brown", "-"),
        "majority_voting_set_8": ("Maj@8", "navy", "-."),
        "confidence_bottom_10_percent_confidence_set_8": ("Conf Bottom 10% @8", "darkred", "-.")
    }

    # Collect values
    metric_values = []
    for key, (label, color, style) in extra_metrics_config.items():
        val = metrics_data.get("baseline", {}).get(key)
        if val is not None:
            metric_values.append({
                'key': key,
                'val': val,
                'label': label,
                'color': color,
                'style': style
            })
            
    # Group by value (with small tolerance)
    from collections import defaultdict
    groups = defaultdict(list)
    for item in metric_values:
        # Round to 6 decimals to detect equality
        val_key = round(item['val'], 6)
        groups[val_key].append(item)
        
    # Plot groups
    for val_key, items in groups.items():
        # Sort items to ensure deterministic layering
        items.sort(key=lambda x: x['key'])
        
        base_lw = 2.5
        step_lw = 3.0
        
        # Draw from widest (background) to thinnest (foreground)
        # If items = [A, B], we want A to be wide, B to be thin (or vice versa).
        # We'll make the first item in the sorted list the widest (background).
        
        for i, item in enumerate(items):
            # Calculate width: widest for i=0, thinnest for i=last
            lw = base_lw + (len(items) - 1 - i) * step_lw
            
            plt.axhline(y=item['val'], color=item['color'], linestyle=item['style'], 
                        linewidth=lw, label=item['label'], alpha=0.9)
        
    plt.title(f'{benchmark_name} Comparison')
    plt.xlabel('Checkpoint')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f'{benchmark_name}_comparison.png')
    plt.savefig(output_path)
    plt.close()
    logger.info(f"Saved comparison plot to {output_path}")

def main():
    base_dir = "/root/projects/Conf_Agg/output_s/outputs/comprehensive_results/Qwen_Qwen3-1.7B_think_True_32768"
    dir1_name = "4000_32_naive_uniform" #baseline yellow line
    dir2_name = "4000_32_baseline" #comparison blue line
    
    dir1 = os.path.join(base_dir, dir1_name)
    dir2 = os.path.join(base_dir, dir2_name)
    
    output_dir = os.path.join(base_dir, "comparison_plots")
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all benchmarks in dir1
    files1 = glob.glob(os.path.join(dir1, "*_accuracy_all_checkpoints.jsonl"))
    benchmarks = set()
    for f in files1:
        filename = os.path.basename(f)
        bench_name = filename.replace('_accuracy_all_checkpoints.jsonl', '')
        benchmarks.add(bench_name)
        
    # Find all benchmarks in dir2
    files2 = glob.glob(os.path.join(dir2, "*_accuracy_all_checkpoints.jsonl"))
    for f in files2:
        filename = os.path.basename(f)
        bench_name = filename.replace('_accuracy_all_checkpoints.jsonl', '')
        benchmarks.add(bench_name)
    
    logger.info(f"Found benchmarks: {benchmarks}")
    
    for bench in benchmarks:
        plot_comparison(bench, dir1, dir2, output_dir, base_dir)

if __name__ == "__main__":
    main()
