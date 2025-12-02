import os
import sys
import glob
import json
import re
import subprocess
import hydra
from omegaconf import DictConfig
from pathlib import Path
import logging

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)

def merge_results_by_benchmark(results_dir):
    """
    Merge checkpoint results into separate files per benchmark.
    Handles baseline deduplication and appends to existing results if present.
    """
    logger.info(f"Starting merge process in {results_dir}")
    
    # Pattern to match the files: {benchmark_name}_accuracy_checkpoint_{num}.jsonl
    pattern = re.compile(r"(.+)_accuracy_checkpoint_(\d+)\.jsonl")
    
    files = os.listdir(results_dir)
    benchmark_groups = {}
    
    for filename in files:
        match = pattern.match(filename)
        if match:
            benchmark_name = match.group(1)
            checkpoint_num = int(match.group(2))
            
            if benchmark_name not in benchmark_groups:
                benchmark_groups[benchmark_name] = []
            
            benchmark_groups[benchmark_name].append((checkpoint_num, filename))
            
    if not benchmark_groups:
        logger.info("No new checkpoint result files found to merge.")
        # We continue to ensure we don't miss anything or if we just want to re-verify, 
        # but primarily we operate on found groups.
    
    for benchmark_name, file_list in benchmark_groups.items():
        # Sort by checkpoint number
        file_list.sort(key=lambda x: x[0])
        
        output_filename = f"{benchmark_name}_accuracy_all_checkpoints.jsonl"
        output_path = os.path.join(results_dir, output_filename)
        
        logger.info(f"Merging {len(file_list)} files for {benchmark_name} into {output_filename}...")
        
        seen_baselines = set()
        baseline_types = [
            "baseline_to_baseline_aggregation",
            "baseline_to_baseline_aggregation_without_confidence"
        ]
        
        existing_data = []
        
        # Read existing data if available
        if os.path.exists(output_path):
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line: continue
                        try:
                            data = json.loads(line)
                            existing_data.append(data)
                            agg_type = data.get("aggregation_type")
                            if agg_type in baseline_types:
                                seen_baselines.add(agg_type)
                        except json.JSONDecodeError:
                            pass
            except Exception as e:
                logger.warning(f"Error reading existing file {output_filename}: {e}")
        
        new_data = []
        
        with open(output_path, 'w', encoding='utf-8') as outfile:
            # First write existing data
            for data in existing_data:
                outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
            
            # Process new files
            for checkpoint_num, filename in file_list:
                file_path = os.path.join(results_dir, filename)
                try:
                    with open(file_path, 'r', encoding='utf-8') as infile:
                        for line in infile:
                            line = line.strip()
                            if not line:
                                continue
                            
                            try:
                                data = json.loads(line)
                                agg_type = data.get("aggregation_type")
                                
                                if agg_type in baseline_types:
                                    # Only write baseline metrics once
                                    if agg_type not in seen_baselines:
                                        outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                                        seen_baselines.add(agg_type)
                                else:
                                    # For other types, add checkpoint info
                                    data["checkpoint"] = checkpoint_num
                                    outfile.write(json.dumps(data, ensure_ascii=False) + "\n")
                                    
                            except json.JSONDecodeError:
                                logger.warning(f"Could not decode JSON in {filename}")
                                continue
                except Exception as e:
                    logger.error(f"Error reading file {filename}: {e}")

        logger.info(f"Finished updating {output_filename}")

        # Delete individual checkpoint files
        for _, filename in file_list:
            try:
                os.remove(os.path.join(results_dir, filename))
            except OSError as e:
                logger.warning(f"Error deleting {filename}: {e}")

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Evaluate multiple checkpoints and then merge the results.
    """
    
    # Setup logging
    log_file = os.path.join(cfg.paths.log_dir, "evaluate_and_merge_checkpoints.log")
    setup_logging(
        log_level=cfg.experiment.get("log_level", "INFO"),
        log_file=log_file
    )
    
    # Get configuration
    eval_config = cfg.evaluation.benchmarks.evaluation
    experiment_name = eval_config.get("experiment_name", "10pct_naive")
    model_dir = cfg.paths.model_dir
    experiment_dir = os.path.join(model_dir, experiment_name)
    
    # Results directory
    results_dir = eval_config.results_dir
    if not os.path.isabs(results_dir):
        results_dir = os.path.join(cfg.paths.output_dir, results_dir)
    
    # Append experiment_name to results_dir
    results_dir = os.path.join(results_dir, experiment_name)
    os.makedirs(results_dir, exist_ok=True)
    
    logger.info(f"Searching for checkpoints in: {experiment_dir}")
    logger.info(f"Results will be saved to: {results_dir}")
    
    if not os.path.exists(experiment_dir):
        logger.error(f"Experiment directory not found: {experiment_dir}")
        return

    # Find all checkpoints
    checkpoints = []
    for name in os.listdir(experiment_dir):
        if name.startswith("checkpoint-") and os.path.isdir(os.path.join(experiment_dir, name)):
            try:
                num = int(name.split("-")[1])
                checkpoints.append(num)
            except ValueError:
                continue
    
    checkpoints.sort()
    
    if not checkpoints:
        logger.warning("No checkpoints found!")
        return
    
    # Filter checkpoints by interval if specified
    checkpoint_interval = eval_config.get("checkpoint_interval", None)
    if checkpoint_interval is not None:
        checkpoint_interval = int(checkpoint_interval)
        original_count = len(checkpoints)
        # Filter: keep checkpoints that are multiples of interval, or the first checkpoint (100)
        filtered_checkpoints = [ckpt for ckpt in checkpoints if ckpt == 100 or ckpt % checkpoint_interval == 0]
        checkpoints = filtered_checkpoints
        logger.info(f"Filtered checkpoints by interval {checkpoint_interval}: {original_count} -> {len(checkpoints)} checkpoints")
        
    logger.info(f"Found {len(checkpoints)} checkpoints: {checkpoints}")
    
    # Check for existing results to resume
    processed_checkpoints = set()
    
    # 1. Check merged files for processed checkpoints
    existing_merged_files = glob.glob(os.path.join(results_dir, "*_accuracy_all_checkpoints.jsonl"))
    for fpath in existing_merged_files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            d = json.loads(line)
                            if 'checkpoint' in d:
                                processed_checkpoints.add(d['checkpoint'])
                        except:
                            pass
        except:
            pass
    
    # 2. Check individual checkpoint files (in case merge hasn't happened yet)
    existing_checkpoint_files = glob.glob(os.path.join(results_dir, "*_accuracy_checkpoint_*.jsonl"))
    if existing_checkpoint_files:
        logger.info(f"Found {len(existing_checkpoint_files)} individual checkpoint result files in {results_dir}")
    for fpath in existing_checkpoint_files:
        # Extract checkpoint number from filename
        # Pattern: {benchmark_name}_accuracy_checkpoint_{num}.jsonl
        filename = os.path.basename(fpath)
        match = re.match(r".+_accuracy_checkpoint_(\d+)\.jsonl", filename)
        if match:
            try:
                ckpt_num = int(match.group(1))
                processed_checkpoints.add(ckpt_num)
            except ValueError:
                pass
            
    if processed_checkpoints:
        logger.info(f"Found previously processed checkpoints: {sorted(list(processed_checkpoints))}")
        checkpoints = [c for c in checkpoints if c not in processed_checkpoints]
        
    if not checkpoints:
        logger.info("All checkpoints have been processed.")
    else:
        logger.info(f"Resuming evaluation for {len(checkpoints)} checkpoints: {checkpoints}")
    
    # Iterate over checkpoints
    for i, ckpt in enumerate(checkpoints):
        logger.info(f"Processing checkpoint {ckpt} ({i+1}/{len(checkpoints)})")
        
        # Additional check: if individual checkpoint files exist, skip (defensive check)
        pattern = os.path.join(results_dir, f"*_accuracy_checkpoint_{ckpt}.jsonl")
        if glob.glob(pattern):
            logger.info(f"Results for checkpoint {ckpt} already exist. Skipping evaluation.")
            continue
        
        # Determine skip_baseline flag
        # Skip baseline for all checkpoints except the first one (100)
        skip_baseline = "true" if ckpt != 100 else "false"
        
        cmd = [
            "python", "scripts/stage4_3_evaluate_aggregation_group_filter.py",
            f"evaluation.benchmarks.evaluation.checkpoint_num={ckpt}",
            f"evaluation.benchmarks.evaluation.experiment_name={experiment_name}",
            f"evaluation.benchmarks.evaluation.skip_baseline={skip_baseline}"
        ]
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Error running evaluation for checkpoint {ckpt}: {e}")
            continue

    # Merge results
    logger.info("Evaluation complete. Merging results...")
    merge_results_by_benchmark(results_dir)
    
    # Plot results
    logger.info("Merging complete. Plotting results...")
    plot_results(results_dir)
    
    logger.info("All tasks completed successfully.")

def plot_results(results_dir):
    """
    Plot the results for each benchmark found in the results directory.
    """
    import matplotlib.pyplot as plt
    
    files = glob.glob(os.path.join(results_dir, "*_accuracy_all_checkpoints.jsonl"))
    
    if not files:
        logger.warning("No merged result files found to plot.")
        return
        
    for file_path in files:
        try:
            plot_benchmark(file_path, plt)
        except Exception as e:
            logger.error(f"Error plotting {file_path}: {e}")

def plot_benchmark(file_path, plt):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    
    if not data:
        logger.warning(f"No data found in {file_path}")
        return

    # Determine benchmark name from filename
    filename = os.path.basename(file_path)
    base_name = filename.replace('_accuracy_all_checkpoints.jsonl', '')
    
    # Search for metrics file
    output_dir = os.path.dirname(file_path)
    metrics_files = glob.glob(os.path.join(output_dir, "*_metrics.json"))
    
    # Also search in parent directory if not found
    if not metrics_files:
        parent_dir = os.path.dirname(output_dir)
        metrics_files = glob.glob(os.path.join(parent_dir, "*_metrics.json"))

    metrics_data = {}
    
    target_metrics_file = None
    for mf in metrics_files:
        if base_name.lower() in os.path.basename(mf).lower():
            target_metrics_file = mf
            break
            
    if target_metrics_file:
        try:
            with open(target_metrics_file, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            logger.info(f"Loaded metrics from {target_metrics_file}")
        except Exception as e:
            logger.warning(f"Failed to load metrics file {target_metrics_file}: {e}")
    
    # Extract Baselines
    baseline_conf = None
    baseline_no_conf = None
    
    # Extract Series
    base_to_agg = []
    agg_to_agg = []
    
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
                base_to_agg.append((ckpt, acc))
        elif agg_type == 'aggllm_to_aggllm_aggregation':
            ckpt = entry.get('checkpoint')
            if ckpt is not None:
                agg_to_agg.append((ckpt, acc))
                
    # Sort by checkpoint
    base_to_agg.sort(key=lambda x: x[0])
    agg_to_agg.sort(key=lambda x: x[0])
    
    # Helper to add extra metrics lines
    def add_extra_metrics(plt_obj):
        extra_metrics = {
            "pass_at_1": ("Pass@1", "purple"),
            "pass_at_8": ("Pass@8", "brown"),
            "majority_voting_set_8": ("Maj@8", "cyan"),
            "confidence_bottom_10_percent_confidence_set_8": ("Conf Bottom 10% @8", "magenta")
        }
        
        for key, (label, color) in extra_metrics.items():
            val = metrics_data.get(key)
            if val is not None:
                 plt_obj.axhline(y=val, color=color, linestyle='-', label=label, alpha=0.7)

    from matplotlib.ticker import FormatStrFormatter

    # Plot 1: Baseline to AggLLM
    if base_to_agg or baseline_conf is not None or baseline_no_conf is not None:
        plt.figure(figsize=(12, 8))
        
        if base_to_agg:
            x, y = zip(*base_to_agg)
            plt.plot(x, y, marker='o', label='AggLLM')
            
        if baseline_conf is not None:
            plt.axhline(y=baseline_conf, color='r', linestyle='--', label='Prompt Agg (with Conf)')
        if baseline_no_conf is not None:
            plt.axhline(y=baseline_no_conf, color='g', linestyle='--', label='Prompt Agg (w/o Conf)')
        
        add_extra_metrics(plt)
            
        plt.title(f'{base_name}')
        plt.xlabel('Checkpoint')
        plt.ylabel('Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        plt.tight_layout()
        
        plot_path1 = os.path.join(output_dir, f'{base_name}_baseline_to_aggllm.png')
        plt.savefig(plot_path1)
        plt.close()
        logger.info(f"Saved plot {plot_path1}")
    
    # Plot 2: AggLLM to AggLLM
    if agg_to_agg:
        plt.figure(figsize=(12, 8))
        
        x, y = zip(*agg_to_agg)
        plt.plot(x, y, marker='s', color='orange', label='AggLLM to AggLLM')
        
        if baseline_conf is not None:
            plt.axhline(y=baseline_conf, color='r', linestyle='--', label='Prompt Agg (with Conf)')
        if baseline_no_conf is not None:
            plt.axhline(y=baseline_no_conf, color='g', linestyle='--', label='Prompt Agg (w/o Conf)')
            
        add_extra_metrics(plt)

        plt.title(f'{base_name}')
        plt.xlabel('Checkpoint')
        plt.ylabel('Accuracy')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.4f'))
        plt.tight_layout()
        
        plot_path2 = os.path.join(output_dir, f'{base_name}_aggllm_to_aggllm.png')
        plt.savefig(plot_path2)
        plt.close()
        logger.info(f"Saved plot {plot_path2}")

if __name__ == "__main__":
    main()
