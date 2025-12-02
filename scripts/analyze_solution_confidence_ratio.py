import os
import json
import re
from collections import defaultdict

def analyze_directory(directory, experiment_label):
    results = defaultdict(lambda: defaultdict(lambda: {
        'total': 0, 
        'has_term_solution': 0, 
        'has_term_confidence': 0,
        'examples': []
    }))
    
    # Regex for terms - Strict matching
    term_solution_pattern = re.compile(r'\bsolutions?\b', re.IGNORECASE)
    term_confidence_pattern = re.compile(r'\bconfidences?\b', re.IGNORECASE)

    # Check if group_size_8 exists, otherwise use the directory itself
    target_path = os.path.join(directory, "group_size_8")
    if not os.path.exists(target_path):
        target_path = directory

    if not os.path.exists(target_path):
        print(f"Warning: {target_path} does not exist. Skipping.")
        return results

    for filename in os.listdir(target_path):
        if filename.endswith(".json") and "aggregation_results_checkpoint_" in filename:
            # Extract benchmark name and checkpoint
            match = re.match(r"(.+)_aggregation_results_checkpoint_(\d+)\.json", filename)
            if not match:
                continue
            
            benchmark_name = match.group(1)
            checkpoint = int(match.group(2))
            
            # Filter range 100-2400
            if not (100 <= checkpoint <= 2400):
                continue
            
            filepath = os.path.join(target_path, filename)
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
            except Exception as e:
                print(f"Error loading {filepath}: {e}")
                continue
            
            if 'baseline_to_aggllm_aggregation' not in data:
                continue
            
            agg_data = data['baseline_to_aggllm_aggregation']
            
            for key, item in agg_data.items():
                if 'prompts' not in item:
                    continue
                
                for prompt in item['prompts']:
                    generated_text = prompt.get('generated_text', '')
                    
                    # Find matches
                    sol_matches = list(term_solution_pattern.finditer(generated_text))
                    conf_matches = list(term_confidence_pattern.finditer(generated_text))
                    
                    has_term_solution = bool(sol_matches)
                    has_term_confidence = bool(conf_matches)
                    
                    results[benchmark_name][checkpoint]['total'] += 1
                    if has_term_solution:
                        results[benchmark_name][checkpoint]['has_term_solution'] += 1
                    if has_term_confidence:
                        results[benchmark_name][checkpoint]['has_term_confidence'] += 1
                        
                    # Collect examples (keep simple for multi-exp to save memory/output space)
                    # Only collect if interesting
                    if has_term_confidence and len(results[benchmark_name][checkpoint]['examples']) < 1:
                         m = conf_matches[0]
                         start = max(0, m.start() - 40)
                         end = min(len(generated_text), m.end() + 40)
                         context = f"...{generated_text[start:end]}..."
                         results[benchmark_name][checkpoint]['examples'].append(context)

    return results

def print_consolidated_results(all_results):
    # Get all benchmarks
    benchmarks = set()
    for exp_res in all_results.values():
        benchmarks.update(exp_res.keys())
    
    for benchmark in sorted(benchmarks):
        print(f"\n{'='*80}")
        print(f"Benchmark: {benchmark}")
        print(f"{'='*80}")
        
        # Print header
        header = f"{'Checkpoint':<10} |"
        for exp_name in all_results.keys():
            header += f" {exp_name[:15]:<15} |"
        print(header)
        print("-" * len(header))
        
        # Get all checkpoints
        checkpoints = set()
        for exp_res in all_results.values():
            if benchmark in exp_res:
                checkpoints.update(exp_res[benchmark].keys())
        
        for checkpoint in sorted(checkpoints):
            row = f"{checkpoint:<10} |"
            for exp_name in all_results.keys():
                res = all_results[exp_name]
                if benchmark in res and checkpoint in res[benchmark]:
                    stats = res[benchmark][checkpoint]
                    total = stats['total']
                    if total > 0:
                        sol_pct = (stats['has_term_solution'] / total) * 100
                        conf_pct = (stats['has_term_confidence'] / total) * 100
                        # Format: Sol% / Conf%
                        val = f"{sol_pct:.1f}% / {conf_pct:.1f}%"
                        row += f" {val:<15} |"
                    else:
                        row += f" {'N/A':<15} |"
                else:
                    row += f" {'-':<15} |"
            print(row)

if __name__ == "__main__":
    experiments = {
        "naive_group8": "/root/projects/Conf_Agg/output_s/outputs/comprehensive_results/Qwen_Qwen3-1.7B_think_True_32768/3200_32_naive",
        "naive_kl0": "/root/projects/Conf_Agg/output_s/outputs/comprehensive_results/Qwen_Qwen3-1.7B_think_True_32768/3200_32_naive_kl0",
        "naive_lr1e-6": "/root/projects/Conf_Agg/output_s/outputs/comprehensive_results/Qwen_Qwen3-1.7B_think_True_32768/3200_32_naive_lr1e-6",
        "baseline_kl0": "/root/projects/Conf_Agg/output_s/outputs/comprehensive_results/Qwen_Qwen3-1.7B_think_True_32768/3200_32_baseline_kl0"
    }
    
    all_results = {}
    for exp_name, exp_path in experiments.items():
        print(f"Analyzing {exp_name}...")
        all_results[exp_name] = analyze_directory(exp_path, exp_name)
        
    print_consolidated_results(all_results)
