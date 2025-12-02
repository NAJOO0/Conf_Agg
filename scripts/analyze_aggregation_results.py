import argparse
import json
import glob
import os
import re
from collections import defaultdict
import statistics
import matplotlib.pyplot as plt
import math

def load_json(filepath):
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        # print(f"Error reading {filepath}: {e}")
        return None

def get_problem_stats(problem_data):
    prompts = problem_data.get('prompts', [])
    if not prompts:
        return 0, 0.0
    count = prompts[0].get('correct_solutions_count', 0)
    correct_count = sum(1 for p in prompts if p.get('is_correct', False))
    success_rate = correct_count / len(prompts)
    return count, success_rate

def get_checkpoints_list(target_num):
    # Get ALL checkpoints from 100 to target with stride 100
    # User requested plotting for ALL checkpoints
    cps = []
    curr = 100
    while curr <= target_num:
        cps.append(curr)
        curr += 100
    return cps

def analyze_file(filepath, target_checkpoint_num):
    print(f"\n{'='*80}")
    print(f"Analyzing File: {filepath}")
    print(f"{'='*80}")
    
    data = load_json(filepath)
    if not data:
        return

    dataset_name = data.get('dataset_name', 'Unknown Dataset')
    print(f"Dataset: {dataset_name}")

    # Determine file pattern
    basename = os.path.basename(filepath)
    match = re.search(r'checkpoint_(\d+)\.json$', basename)
    if not match:
        print("Could not parse checkpoint number from filename.")
        return
    
    prefix = basename[:match.start(1)] # includes 'checkpoint_'
    suffix = ".json"
    dir_path = os.path.dirname(filepath)
    
    # Identify checkpoints to load
    target_num_int = int(target_checkpoint_num)
    checkpoints_to_load = get_checkpoints_list(target_num_int)
    
    # Load Checkpoint Data
    cp_lookup = {}
    
    for cp in checkpoints_to_load:
        cp_filename = f"{prefix}{cp}{suffix}"
        cp_path = os.path.join(dir_path, cp_filename)
        
        if os.path.exists(cp_path):
            cdata = load_json(cp_path)
            if cdata:
                # Extract AggLLM accuracy (prioritize AggLLM)
                ref_keys = ['baseline_to_aggllm_aggregation', 'baseline_to_baseline_aggregation', 'baseline_to_baseline_aggregation_without_confidence']
                ref_key = next((k for k in ref_keys if k in cdata), None)
                
                if ref_key:
                    lookup = {}
                    for pid, pdata in cdata[ref_key].items():
                        _, acc = get_problem_stats(pdata)
                        lookup[pid] = acc
                    cp_lookup[cp] = lookup

    # Methods from Target File
    methods = [
        ('baseline_to_baseline_aggregation_without_confidence', 'Base(NoConf)'),
        ('baseline_to_baseline_aggregation', 'Base(Conf)')
    ]
    
    valid_methods = []
    for k, label in methods:
        if k in data:
            valid_methods.append((k, label))

    # Stats Aggregation
    stats = defaultdict(lambda: {
        'occurrences': 0,
        'method_sums': [0.0] * len(valid_methods),
        'cp_sums': {cp: 0.0 for cp in checkpoints_to_load}
    })
    
    if valid_methods:
        problems = data[valid_methods[0][0]]
    else:
        any_key = next((k for k in data.keys() if 'aggregation' in k), None)
        if not any_key: return
        problems = data[any_key]

    for pid, problem in problems.items():
        count, _ = get_problem_stats(problem)
        prompts = problem.get('prompts', [])
        n_inst = len(prompts)
        
        stats[count]['occurrences'] += n_inst
        
        for i, (m_key, _) in enumerate(valid_methods):
            if pid in data[m_key]:
                p_data = data[m_key][pid]
                p_prompts = p_data.get('prompts', [])
                # Count actual correct instances
                p_corr = sum(1 for p in p_prompts if p.get('is_correct', False))
                stats[count]['method_sums'][i] += p_corr
        
        for cp in checkpoints_to_load:
            if cp in cp_lookup and pid in cp_lookup[cp]:
                # cp_lookup stores success_rate
                # Add equivalent correct count: rate * n_inst
                acc = cp_lookup[cp][pid]
                stats[count]['cp_sums'][cp] += acc * n_inst

    # Print Full Table
    # Columns: Count, Occur, [Methods], [CPs]
    headers = ['Count', 'Occur.'] + [label for _, label in valid_methods] + [f"CP{cp}" for cp in checkpoints_to_load]
    
    col_widths = [5, 8] + [12] * len(valid_methods) + [8] * len(checkpoints_to_load)
    
    header_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths])
    row_fmt = " | ".join([f"{{:<{w}}}" for w in col_widths[:2]]) + " | " + " | ".join([f"{{:<{w}.4f}}" for w in col_widths[2:]])
    
    print("\n[Full Table]")
    print(header_fmt.format(*headers))
    print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    
    sorted_counts = sorted(stats.keys())
    
    grand_totals = {
        'occurrences': 0,
        'method_sums': [0.0] * len(valid_methods),
        'cp_sums': {cp: 0.0 for cp in checkpoints_to_load}
    }
    
    # Data for plotting
    plot_data = {} # count -> { 'cps': [], 'agg_accs': [], 'base_noconf': val, 'base_conf': val }
    
    for c in sorted_counts:
        s = stats[c]
        occ = s['occurrences']
        if occ == 0: continue
        
        method_avgs = [ms / occ for ms in s['method_sums']]
        cp_avgs = [s['cp_sums'][cp] / occ for cp in checkpoints_to_load]
        
        print(row_fmt.format(c, occ, *method_avgs, *cp_avgs))
        
        # Collect plot data
        plot_data[c] = {
            'cps': checkpoints_to_load,
            'agg_accs': cp_avgs,
            'base_noconf': method_avgs[0] if len(method_avgs) > 0 else 0,
            'base_conf': method_avgs[1] if len(method_avgs) > 1 else 0
        }
        
        grand_totals['occurrences'] += occ
        for i in range(len(valid_methods)):
            grand_totals['method_sums'][i] += s['method_sums'][i]
        for cp in checkpoints_to_load:
            grand_totals['cp_sums'][cp] += s['cp_sums'][cp]
            
    # Summary
    print("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)))
    if grand_totals['occurrences'] > 0:
        occ = grand_totals['occurrences']
        method_avgs = [ms / occ for ms in grand_totals['method_sums']]
        cp_avgs = [grand_totals['cp_sums'][cp] / occ for cp in checkpoints_to_load]
        print(row_fmt.format("Total", occ, *method_avgs, *cp_avgs))
        
        plot_data['Total'] = {
            'cps': checkpoints_to_load,
            'agg_accs': cp_avgs,
            'base_noconf': method_avgs[0] if len(method_avgs) > 0 else 0,
            'base_conf': method_avgs[1] if len(method_avgs) > 1 else 0
        }

    # Simplified Table
    # Columns: Count, Occur, Base(NoConf), Base(Conf), CP{Target} (AggLLM)
    target_cp_str = str(target_checkpoint_num)
    # Find index of target cp in checkpoints_to_load
    try:
        target_idx = checkpoints_to_load.index(int(target_checkpoint_num))
    except ValueError:
        target_idx = -1
        
    print("\n[Simplified Table]")
    headers_sim = ['Count', 'Occur.', 'Base(NoConf)', 'Base(Conf)', f'CP{target_checkpoint_num}(Agg)']
    col_widths_sim = [5, 8, 12, 12, 15]
    header_fmt_sim = " | ".join([f"{{:<{w}}}" for w in col_widths_sim])
    row_fmt_sim = " | ".join([f"{{:<{w}}}" for w in col_widths_sim[:2]]) + " | " + " | ".join([f"{{:<{w}.4f}}" for w in col_widths_sim[2:]])
    
    print(header_fmt_sim.format(*headers_sim))
    print("-" * (sum(col_widths_sim) + 3 * (len(col_widths_sim) - 1)))
    
    for c in sorted_counts:
        s = stats[c]
        occ = s['occurrences']
        if occ == 0: continue
        
        method_avgs = [ms / occ for ms in s['method_sums']]
        cp_avgs = [s['cp_sums'][cp] / occ for cp in checkpoints_to_load]
        
        # Base NoConf, Base Conf, Target CP Agg
        vals = [
            method_avgs[0] if len(method_avgs) > 0 else 0,
            method_avgs[1] if len(method_avgs) > 1 else 0,
            cp_avgs[target_idx] if target_idx != -1 else 0
        ]
        print(row_fmt_sim.format(c, occ, *vals))
        
    if grand_totals['occurrences'] > 0:
        print("-" * (sum(col_widths_sim) + 3 * (len(col_widths_sim) - 1)))
        occ = grand_totals['occurrences']
        method_avgs = [ms / occ for ms in grand_totals['method_sums']]
        cp_avgs = [grand_totals['cp_sums'][cp] / occ for cp in checkpoints_to_load]
        vals = [
            method_avgs[0] if len(method_avgs) > 0 else 0,
            method_avgs[1] if len(method_avgs) > 1 else 0,
            cp_avgs[target_idx] if target_idx != -1 else 0
        ]
        print(row_fmt_sim.format("Total", occ, *vals))

    # Plotting
    if plot_data:
        # Determine grid size
        keys = sorted([k for k in plot_data.keys() if k != 'Total']) + ['Total']
        n_plots = len(keys)
        cols = 3
        rows = math.ceil(n_plots / cols)
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
        axes = axes.flatten()
        
        for i, key in enumerate(keys):
            ax = axes[i]
            d = plot_data[key]
            
            # Plot AggLLM Trajectory
            ax.plot(d['cps'], d['agg_accs'], marker='o', label='AggLLM')
            
            # Plot Baselines
            ax.axhline(y=d['base_noconf'], color='r', linestyle='--', label='Base(NoConf)')
            ax.axhline(y=d['base_conf'], color='g', linestyle='--', label='Base(Conf)')
            
            ax.set_title(f"Count: {key}")
            ax.set_xlabel("Checkpoint")
            ax.set_ylabel("Accuracy")
            ax.grid(True)
            ax.legend()
            
        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')
            
        plt.tight_layout()
        
        # Save plot
        output_dir = os.path.join(os.path.dirname(filepath), 'analysis_plots')
        os.makedirs(output_dir, exist_ok=True)
        plot_filename = f"{dataset_name.replace('/', '_')}_trajectory.png"
        save_path = os.path.join(output_dir, plot_filename)
        plt.savefig(save_path)
        print(f"\nPlot saved to: {save_path}")
        plt.close()
        
        # Save Full Table to File
        full_table_filename = f"{dataset_name.replace('/', '_')}_full_table.txt"
        full_table_path = os.path.join(output_dir, full_table_filename)
        with open(full_table_path, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(header_fmt.format(*headers) + "\n")
            f.write("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)) + "\n")
            
            for c in sorted_counts:
                s = stats[c]
                occ = s['occurrences']
                if occ == 0: continue
                
                method_avgs = [ms / occ for ms in s['method_sums']]
                cp_avgs = [s['cp_sums'][cp] / occ for cp in checkpoints_to_load]
                
                f.write(row_fmt.format(c, occ, *method_avgs, *cp_avgs) + "\n")
            
            if grand_totals['occurrences'] > 0:
                f.write("-" * (sum(col_widths) + 3 * (len(col_widths) - 1)) + "\n")
                occ = grand_totals['occurrences']
                method_avgs = [ms / occ for ms in grand_totals['method_sums']]
                cp_avgs = [grand_totals['cp_sums'][cp] / occ for cp in checkpoints_to_load]
                f.write(row_fmt.format("Total", occ, *method_avgs, *cp_avgs) + "\n")
        print(f"Full table saved to: {full_table_path}")
        
        # Save Simplified Table to File
        simple_table_filename = f"{dataset_name.replace('/', '_')}_simplified_table.txt"
        simple_table_path = os.path.join(output_dir, simple_table_filename)
        with open(simple_table_path, 'w') as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(header_fmt_sim.format(*headers_sim) + "\n")
            f.write("-" * (sum(col_widths_sim) + 3 * (len(col_widths_sim) - 1)) + "\n")
            
            for c in sorted_counts:
                s = stats[c]
                occ = s['occurrences']
                if occ == 0: continue
                
                method_avgs = [ms / occ for ms in s['method_sums']]
                cp_avgs = [s['cp_sums'][cp] / occ for cp in checkpoints_to_load]
                
                vals = [
                    method_avgs[0] if len(method_avgs) > 0 else 0,
                    method_avgs[1] if len(method_avgs) > 1 else 0,
                    cp_avgs[target_idx] if target_idx != -1 else 0
                ]
                f.write(row_fmt_sim.format(c, occ, *vals) + "\n")
            
            if grand_totals['occurrences'] > 0:
                f.write("-" * (sum(col_widths_sim) + 3 * (len(col_widths_sim) - 1)) + "\n")
                occ = grand_totals['occurrences']
                method_avgs = [ms / occ for ms in grand_totals['method_sums']]
                cp_avgs = [grand_totals['cp_sums'][cp] / occ for cp in checkpoints_to_load]
                vals = [
                    method_avgs[0] if len(method_avgs) > 0 else 0,
                    method_avgs[1] if len(method_avgs) > 1 else 0,
                    cp_avgs[target_idx] if target_idx != -1 else 0
                ]
                f.write(row_fmt_sim.format("Total", occ, *vals) + "\n")
        print(f"Simplified table saved to: {simple_table_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze aggregation results for a specific checkpoint.")
    parser.add_argument('--checkpoint_num', type=str, required=True, help='Checkpoint number to analyze')
    parser.add_argument('--base_dir', type=str, default='/root/projects/Conf_Agg/output_s/outputs/comprehensive_results/Qwen_Qwen3-1.7B_think_True_32768/3200_32_baseline_kl0', help='Base directory to search for files')
    args = parser.parse_args()
    
    print(f"Searching for files with checkpoint {args.checkpoint_num} in {args.base_dir}...")
    
    search_pattern = os.path.join(args.base_dir, '**', f'*aggregation_results_checkpoint_{args.checkpoint_num}.json')
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"No files found matching pattern: {search_pattern}")
        return
        
    print(f"Found {len(files)} files.")
    for f in files:
        analyze_file(f, args.checkpoint_num)

if __name__ == "__main__":
    main()
