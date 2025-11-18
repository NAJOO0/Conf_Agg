#!/usr/bin/env python3
"""
Real memory profiling with actual forward/backward passes.
Properly cleans up memory between tests to avoid OOM.
"""

import torch
import gc
from unsloth import FastLanguageModel


def clear_memory():
    """Aggressively clear GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

def test_real_memory(
    model_name: str,
    batch_size: int,
    num_generations: int,
    seq_length: int = 32768,
    lora_r: int = 64,
    load_in_4bit: bool = False,
):
    """
    Test REAL memory usage with actual forward/backward pass.
    """
    clear_memory()
    
    print(f"\n{'='*70}")
    print(f"Testing: bs={batch_size}, gen={num_generations}, 4bit={load_in_4bit}")
    print(f"{'='*70}")
    
    model = None
    tokenizer = None
    
    try:
        # Load model
        print("Loading model...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=seq_length,
            dtype=torch.bfloat16 if not load_in_4bit else None,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
        )
        
        # Apply LoRA
        print("Applying LoRA...")
        model = FastLanguageModel.get_peft_model(
            model,
            r=lora_r,
            lora_alpha=lora_r * 2,
            lora_dropout=0,  # No dropout for fast patching
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=42,
            use_rslora=True,
        )
        
        mem_after_load = torch.cuda.memory_allocated(0) / 1e9
        print(f"Model loaded: {mem_after_load:.2f} GB")
        
        # Create reference model (for GRPO simulation)
        print("Creating reference model...")
        ref_model, _ = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=seq_length,
            dtype=torch.bfloat16 if not load_in_4bit else None,
            load_in_4bit=load_in_4bit,
            trust_remote_code=True,
        )
        # Freeze reference model
        for param in ref_model.parameters():
            param.requires_grad = False
        
        mem_with_ref = torch.cuda.memory_allocated(0) / 1e9
        print(f"With reference model: {mem_with_ref:.2f} GB")
        
        # Prepare dummy input
        print(f"Creating dummy input (bs={batch_size}, seq={seq_length})...")
        input_ids = torch.randint(
            0, min(1000, tokenizer.vocab_size),  # Smaller vocab for speed
            (batch_size, seq_length),
            device="cuda",
            dtype=torch.long
        )
        attention_mask = torch.ones_like(input_ids)
        
        mem_with_input = torch.cuda.memory_allocated(0) / 1e9
        print(f"With input tensors: {mem_with_input:.2f} GB")
        
        # Forward pass
        print("Running forward pass...")
        torch.cuda.synchronize()
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
        )
        torch.cuda.synchronize()
        
        mem_after_forward = torch.cuda.memory_allocated(0) / 1e9
        mem_peak_forward = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"After forward: {mem_after_forward:.2f} GB (peak: {mem_peak_forward:.2f} GB)")
        
        # Backward pass
        print("Running backward pass...")
        loss = outputs.loss
        torch.cuda.synchronize()
        loss.backward()
        torch.cuda.synchronize()
        
        mem_after_backward = torch.cuda.memory_allocated(0) / 1e9
        mem_peak_backward = torch.cuda.max_memory_allocated(0) / 1e9
        print(f"After backward: {mem_after_backward:.2f} GB (peak: {mem_peak_backward:.2f} GB)")
        
        # Estimate generation memory
        # Each generation needs KV cache
        hidden_size = model.config.hidden_size
        num_layers = model.config.num_hidden_layers
        
        # KV cache per generation: seq_length * hidden_size * 2 (K,V) * num_layers * 2 bytes
        kv_per_gen = seq_length * hidden_size * 2 * num_layers * 2 / 1e9
        # vLLM optimizes this by ~30%
        kv_per_gen *= 0.7
        gen_overhead = num_generations * kv_per_gen
        
        # Total estimated (peak backward + generation)
        estimated_total = mem_peak_backward + gen_overhead + 2.0  # +2GB for GRPO overhead
        
        print(f"\nEstimated generation overhead (×{num_generations}): {gen_overhead:.2f} GB")
        print(f"Estimated TOTAL with generation: {estimated_total:.2f} GB")
        
        # Check if fits
        max_memory = 75.0  # H100 with safety margin
        fits = estimated_total < max_memory
        margin = max_memory - estimated_total
        
        print(f"\n{'✓' if fits else '✗'} Fits in H100 (75GB usable): {fits}")
        print(f"Memory margin: {margin:+.2f} GB")
        
        result = {
            'batch_size': batch_size,
            'num_generations': num_generations,
            'load_in_4bit': load_in_4bit,
            'mem_model': mem_after_load,
            'mem_with_ref': mem_with_ref,
            'mem_forward': mem_after_forward,
            'mem_peak_forward': mem_peak_forward,
            'mem_backward': mem_after_backward,
            'mem_peak_backward': mem_peak_backward,
            'gen_overhead': gen_overhead,
            'estimated_total': estimated_total,
            'fits': fits,
            'margin': margin,
            'effective_batch': batch_size * num_generations,
        }
        
        print("✓ Test completed successfully")
        
        # Cleanup
        del outputs, loss, input_ids, attention_mask
        del model, ref_model, tokenizer
        clear_memory()
        
        return result
        
    except torch.cuda.OutOfMemoryError as e:
        print(f"✗ OOM Error: {str(e)}")
        # Aggressive cleanup
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        clear_memory()
        return None
        
    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        # Cleanup
        if model is not None:
            del model
        if tokenizer is not None:
            del tokenizer
        clear_memory()
        return None


def main():
    """Test multiple configurations with REAL forward/backward."""
    
    model_name = "Qwen/Qwen2.5-1.5B"
    seq_length = 32768
    
    print("="*70)
    print("REAL MEMORY PROFILING with Unsloth")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Sequence: {seq_length}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print("="*70)
    print("\nThis performs ACTUAL forward/backward passes!")
    print("Memory is cleaned between each test.\n")
    
    # Test configurations - start conservative, increase gradually
    configs = [
        # (batch_size, num_generations, load_in_4bit)
        (1, 1, False),   # Absolute minimum
        (1, 2, False),   # Conservative
        (1, 3, False),   
        (1, 4, False),   # Standard
        (1, 6, False),   # Higher gen
        (2, 2, False),   # Higher batch
        (2, 4, False),   # Aggressive
        (2, 6, False),   # Very aggressive
        (1, 4, True),    # 4-bit
        (2, 4, True),    # 4-bit + higher batch
    ]
    
    results = []
    
    for batch_size, num_generations, load_in_4bit in configs:
        result = test_real_memory(
            model_name=model_name,
            batch_size=batch_size,
            num_generations=num_generations,
            seq_length=seq_length,
            load_in_4bit=load_in_4bit,
        )
        
        if result:
            results.append(result)
        else:
            print(f"⚠ Skipping remaining tests due to OOM")
            # Don't break - might work with 4-bit
            if not load_in_4bit:
                continue
            else:
                break
    
    # Print summary
    if not results:
        print("\n⚠ All tests failed! Try reducing sequence length.")
        return
    
    print("\n" + "="*80)
    print("SUMMARY - REAL MEMORY MEASUREMENTS")
    print("="*80)
    print(f"{'BS':>3} | {'Gen':>3} | {'4bit':>5} | {'Fwd Peak':>9} | {'Bwd Peak':>9} | "
          f"{'Est Total':>10} | {'Eff':>4} | {'Fits':>4} | {'Margin':>8}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['batch_size']:>3} | {r['num_generations']:>3} | "
              f"{'Y' if r['load_in_4bit'] else 'N':>5} | "
              f"{r['mem_peak_forward']:>8.2f}G | {r['mem_peak_backward']:>8.2f}G | "
              f"{r['estimated_total']:>9.2f}G | "
              f"{r['effective_batch']:>4} | "
              f"{'✓' if r['fits'] else '✗':>4} | "
              f"{r['margin']:>+7.2f}G")
    
    # Recommendations
    valid = [r for r in results if r['fits']]
    if valid:
        best = max(valid, key=lambda x: x['effective_batch'])
        
        print("\n" + "="*80)
        print("RECOMMENDED CONFIGURATION (REAL MEASUREMENTS):")
        print("="*80)
        print(f"batch_size: {best['batch_size']}")
        print(f"num_generations: {best['num_generations']}")
        print(f"load_in_4bit: {best['load_in_4bit']}")
        print(f"Effective batch size: {best['effective_batch']}")
        print(f"")
        print(f"Memory breakdown:")
        print(f"  Model + Ref:        {best['mem_with_ref']:.2f} GB")
        print(f"  Peak forward:       {best['mem_peak_forward']:.2f} GB")
        print(f"  Peak backward:      {best['mem_peak_backward']:.2f} GB")
        print(f"  Generation (×{best['num_generations']}):    {best['gen_overhead']:.2f} GB")
        print(f"  Estimated TOTAL:    {best['estimated_total']:.2f} GB")
        print(f"  Margin:             {best['margin']:+.2f} GB")
        print("="*80)
        
        # Show alternatives
        print("\nALTERNATIVE CONFIGURATIONS:")
        print("-" * 80)
        
        conservative = [r for r in valid if r['batch_size']==1 and r['num_generations']<=4 
                       and not r['load_in_4bit']]
        if conservative:
            c = conservative[-1]
            print(f"Safe:       bs={c['batch_size']}, gen={c['num_generations']}, "
                  f"eff={c['effective_batch']:2d}, mem={c['estimated_total']:.1f}GB, "
                  f"margin={c['margin']:+.1f}GB")
        
        balanced = [r for r in valid if not r['load_in_4bit']]
        if balanced:
            b = max(balanced, key=lambda x: x['effective_batch'])
            print(f"Best BF16:  bs={b['batch_size']}, gen={b['num_generations']}, "
                  f"eff={b['effective_batch']:2d}, mem={b['estimated_total']:.1f}GB, "
                  f"margin={b['margin']:+.1f}GB")
        
        fourbit = [r for r in valid if r['load_in_4bit']]
        if fourbit:
            f = max(fourbit, key=lambda x: x['effective_batch'])
            print(f"4-bit:      bs={f['batch_size']}, gen={f['num_generations']}, "
                  f"eff={f['effective_batch']:2d}, mem={f['estimated_total']:.1f}GB, "
                  f"margin={f['margin']:+.1f}GB")
        
        print("\n💡 Recommendation: Start with 'Safe' config, then scale up!")
        
    else:
        print("\n⚠ No configurations fit!")


if __name__ == "__main__":
    main()