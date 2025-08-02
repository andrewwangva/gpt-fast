#!/usr/bin/env python3
"""
Entropy Diagnostics Tool for Normal Decoding

Performs normal (non-speculative) decoding while tracking and displaying 
the entropy of each generated token to understand model uncertainty patterns.
"""

import sys
import argparse
import torch
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json
import numpy as np

# Import from the existing generate.py
from generate import (
    Transformer, get_tokenizer, encode_tokens, _load_model, 
    logits_to_probs, multinomial_sample_one_no_sync, 
    default_device, prefill, decode_one_token, create_block_mask, causal_mask
)

def entropy(probs: torch.Tensor) -> torch.Tensor:
    """Calculate entropy of probability distribution"""
    eps = 1e-10
    return -torch.sum(probs * torch.log(probs + eps), dim=-1)

def top_k_entropy(probs: torch.Tensor, k: int = 10) -> float:
    """Calculate entropy considering only top-k most likely tokens"""
    top_probs, _ = torch.topk(probs, k, dim=-1)
    # Renormalize
    top_probs = top_probs / top_probs.sum(dim=-1, keepdim=True)
    return entropy(top_probs).item()

def diagnostic_decode_with_entropy(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    tokenizer,
    **sampling_kwargs
) -> Tuple[torch.Tensor, List[Dict]]:
    """
    Normal decoding with detailed entropy tracking for each token
    """
    device = prompt.device
    batch_size = 1
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    max_seq_length = min(T_new, model.config.block_size)
    
    # Setup model caches
    with torch.device(device):
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
    
    # Initialize sequence tensor
    seq = torch.empty(batch_size, T_new, dtype=prompt.dtype, device=device)
    seq[:, :T] = prompt.view(1, -1)
    
    # Prefill phase
    print(f"🚀 Starting generation with entropy diagnostics...")
    print(f"📝 Prompt: '{tokenizer.decode(prompt.tolist())}'")
    print(f"🔢 Prompt length: {T} tokens")
    print(f"🎯 Generating {max_new_tokens} new tokens")
    print(f"{'='*100}")
    
    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()
    seq[:, T] = next_token.squeeze()
    
    # Track diagnostics for each generated token
    token_diagnostics = []
    
    # Generation loop with entropy tracking
    print(f"{'Step':<4} {'Pos':<4} {'Token':<20} {'Entropy':<8} {'Top5_Ent':<9} {'Top10_Ent':<10} {'MaxProb':<8} {'Top5_Tokens':<40}")
    print(f"{'-'*105}")
    
    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    block_mask = create_block_mask(causal_mask, 1, 1, model.max_seq_length, model.max_seq_length, device=device)
    
    for step in range(max_new_tokens - 1):
        # Get logits for current token
        logits = model(next_token.view(batch_size, -1), input_pos)
        
        # Convert to probabilities
        probs = logits_to_probs(logits[:, -1], **sampling_kwargs)  # [batch_size, vocab_size]
        
        # Calculate various entropy measures
        full_entropy = entropy(probs[0]).item()
        top5_entropy = top_k_entropy(probs[0], k=5)
        top10_entropy = top_k_entropy(probs[0], k=10)
        
        # Get max probability and top tokens for display
        max_prob = probs[0].max().item()
        top5_probs, top5_indices = torch.topk(probs[0], 5)
        top5_tokens = [tokenizer.decode([idx.item()]) for idx in top5_indices]
        top5_str = " | ".join([f"'{tok}'" for tok in top5_tokens[:3]])  # Show first 3
        
        # Sample next token
        next_token = multinomial_sample_one_no_sync(probs)
        current_token_str = tokenizer.decode([next_token[0].item()])
        
        # Store diagnostics
        token_info = {
            'step': step,
            'position': T + step + 1,
            'token_id': next_token[0].item(),
            'token_str': current_token_str,
            'full_entropy': full_entropy,
            'top5_entropy': top5_entropy,
            'top10_entropy': top10_entropy,
            'max_prob': max_prob,
            'top5_tokens': top5_tokens,
            'top5_probs': top5_probs.tolist()
        }
        token_diagnostics.append(token_info)
        
        # Display current token info
        print(f"{step:<4} {T + step + 1:<4} '{current_token_str}':<18 {full_entropy:<8.3f} {top5_entropy:<9.3f} {top10_entropy:<10.3f} {max_prob:<8.3f} {top5_str:<40}")
        
        # Update sequence and position
        seq[:, T + step + 1] = next_token[0]
        input_pos += 1
        
        # Stop if we hit EOS token
        if next_token[0].item() == tokenizer.eos_id():
            print(f"🛑 Hit EOS token, stopping generation")
            break
    
    return seq, token_diagnostics

def analyze_entropy_patterns(diagnostics: List[Dict], tokenizer) -> Dict:
    """Analyze entropy patterns and provide insights"""
    if not diagnostics:
        return {}
    
    entropies = [d['full_entropy'] for d in diagnostics]
    top5_entropies = [d['top5_entropy'] for d in diagnostics]
    top10_entropies = [d['top10_entropy'] for d in diagnostics]
    max_probs = [d['max_prob'] for d in diagnostics]
    
    # Basic statistics
    stats = {
        'total_tokens': len(diagnostics),
        'avg_entropy': np.mean(entropies),
        'std_entropy': np.std(entropies),
        'min_entropy': np.min(entropies),
        'max_entropy': np.max(entropies),
        'median_entropy': np.median(entropies),
        'avg_top5_entropy': np.mean(top5_entropies),
        'avg_top10_entropy': np.mean(top10_entropies),
        'avg_max_prob': np.mean(max_probs)
    }
    
    # Entropy percentiles
    percentiles = [10, 20, 25, 50, 75, 80, 90, 95]
    entropy_percentiles = {}
    for p in percentiles:
        entropy_percentiles[f'p{p}'] = np.percentile(entropies, p)
    
    stats['entropy_percentiles'] = entropy_percentiles
    
    # High/low entropy token analysis
    cutoff_20_percent = stats['max_entropy'] * 0.2
    cutoff_80_percent = np.percentile(entropies, 80)
    
    high_entropy_tokens = [d for d in diagnostics if d['full_entropy'] >= cutoff_80_percent]
    low_entropy_tokens = [d for d in diagnostics if d['full_entropy'] <= cutoff_20_percent]
    
    stats['cutoff_analysis'] = {
        'cutoff_20_percent_of_max': cutoff_20_percent,
        'cutoff_80th_percentile': cutoff_80_percent,
        'tokens_above_80th_percentile': len(high_entropy_tokens),
        'tokens_below_20_percent_of_max': len(low_entropy_tokens),
        'high_entropy_examples': high_entropy_tokens[:5],  # First 5 examples
        'low_entropy_examples': low_entropy_tokens[:5]
    }
    
    return stats

def print_entropy_analysis(stats: Dict):
    """Print detailed entropy analysis"""
    print(f"\n{'='*80}")
    print(f"📊 ENTROPY ANALYSIS RESULTS")
    print(f"{'='*80}")
    
    print(f"\n📈 BASIC STATISTICS:")
    print(f"   🔢 Total tokens generated: {stats['total_tokens']}")
    print(f"   📊 Average entropy: {stats['avg_entropy']:.3f}")
    print(f"   📏 Standard deviation: {stats['std_entropy']:.3f}")
    print(f"   ⬇️  Minimum entropy: {stats['min_entropy']:.3f}")
    print(f"   ⬆️  Maximum entropy: {stats['max_entropy']:.3f}")
    print(f"   🎯 Median entropy: {stats['median_entropy']:.3f}")
    print(f"   🔝 Avg top-5 entropy: {stats['avg_top5_entropy']:.3f}")
    print(f"   🔟 Avg top-10 entropy: {stats['avg_top10_entropy']:.3f}")
    print(f"   💪 Avg max probability: {stats['avg_max_prob']:.3f}")
    
    print(f"\n📊 ENTROPY PERCENTILES:")
    for p, val in stats['entropy_percentiles'].items():
        print(f"   {p.upper()}: {val:.3f}")
    
    cutoff_info = stats['cutoff_analysis']
    print(f"\n🎯 CUTOFF ANALYSIS:")
    print(f"   📏 20% of max entropy cutoff: {cutoff_info['cutoff_20_percent_of_max']:.3f}")
    print(f"   📊 80th percentile cutoff: {cutoff_info['cutoff_80th_percentile']:.3f}")
    print(f"   ⬆️  Tokens above 80th percentile: {cutoff_info['tokens_above_80th_percentile']}")
    print(f"   ⬇️  Tokens below 20% of max: {cutoff_info['tokens_below_20_percent_of_max']}")
    
    if cutoff_info['high_entropy_examples']:
        print(f"\n🔥 HIGH ENTROPY EXAMPLES (80th percentile+):")
        for i, token in enumerate(cutoff_info['high_entropy_examples']):
            print(f"   {i+1}. Step {token['step']}: '{token['token_str']}' (entropy: {token['full_entropy']:.3f})")
    
    if cutoff_info['low_entropy_examples']:
        print(f"\n❄️  LOW ENTROPY EXAMPLES (20% of max):")
        for i, token in enumerate(cutoff_info['low_entropy_examples']):
            print(f"   {i+1}. Step {token['step']}: '{token['token_str']}' (entropy: {token['full_entropy']:.3f})")
    
    # Insights and recommendations
    print(f"\n💡 INSIGHTS:")
    avg_ent = stats['avg_entropy']
    if avg_ent < 2.0:
        print(f"   🎯 Low average entropy ({avg_ent:.2f}) suggests the model is quite confident")
    elif avg_ent > 4.0:
        print(f"   🤔 High average entropy ({avg_ent:.2f}) suggests significant uncertainty")
    else:
        print(f"   ⚖️  Moderate entropy ({avg_ent:.2f}) indicates balanced confidence")
    
    high_entropy_pct = (cutoff_info['tokens_above_80th_percentile'] / stats['total_tokens']) * 100
    print(f"   📊 {high_entropy_pct:.1f}% of tokens have high uncertainty (80th percentile+)")
    
    if stats['std_entropy'] > 1.0:
        print(f"   📈 High entropy variance ({stats['std_entropy']:.2f}) suggests mixed confidence across tokens")
    else:
        print(f"   📉 Low entropy variance ({stats['std_entropy']:.2f}) suggests consistent confidence")

def main():
    parser = argparse.ArgumentParser(description='Entropy Diagnostics for Normal Decoding')
    parser.add_argument('--checkpoint_path', type=Path, required=True)
    parser.add_argument('--prompt', type=str, default="The quick brown fox")
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=200)
    parser.add_argument('--device', type=str, default=default_device)
    parser.add_argument('--save_diagnostics', type=str, help='Save diagnostics to JSON file')
    
    args = parser.parse_args()
    
    print("🔧 Loading model...")
    
    # Load model
    precision = torch.bfloat16
    model = _load_model(args.checkpoint_path, args.device, precision, use_tp=False)
    
    # Load tokenizer
    tokenizer_path = args.checkpoint_path.parent / "tokenizer.json"
    tokenizer = get_tokenizer(tokenizer_path, args.checkpoint_path)
    
    # Encode prompt
    encoded = encode_tokens(tokenizer, args.prompt, bos=True, device=args.device)
    
    print(f"✅ Model loaded!")
    print(f"📝 Original prompt: '{args.prompt}'")
    print(f"🔢 Encoded length: {encoded.size(-1)} tokens")
    
    # Perform generation with entropy diagnostics
    start_time = time.time()
    seq, diagnostics = diagnostic_decode_with_entropy(
        model=model,
        prompt=encoded,
        max_new_tokens=args.max_new_tokens,
        tokenizer=tokenizer,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    generation_time = time.time() - start_time
    
    # Display full generated text
    print(f"\n{'='*80}")
    print(f"📜 COMPLETE GENERATED TEXT:")
    print(f"{'='*80}")
    generated_text = tokenizer.decode(seq[0].tolist())
    print(generated_text)
    print(f"{'='*80}")
    
    print(f"\n⏱️  Generation time: {generation_time:.2f} seconds")
    print(f"🚀 Generation speed: {len(diagnostics) / generation_time:.2f} tokens/sec")
    
    # Analyze entropy patterns
    if diagnostics:
        stats = analyze_entropy_patterns(diagnostics, tokenizer)
        print_entropy_analysis(stats)
        
        # Save diagnostics if requested
        if args.save_diagnostics:
            save_data = {
                'metadata': {
                    'prompt': args.prompt,
                    'max_new_tokens': args.max_new_tokens,
                    'temperature': args.temperature,
                    'top_k': args.top_k,
                    'generation_time': generation_time,
                    'generated_text': generated_text
                },
                'token_diagnostics': diagnostics,
                'entropy_analysis': stats
            }
            
            with open(args.save_diagnostics, 'w') as f:
                json.dump(save_data, f, indent=2, default=lambda x: float(x) if torch.is_tensor(x) else x)
            print(f"\n💾 Diagnostics saved to {args.save_diagnostics}")
    
    else:
        print("⚠️  No tokens were generated for analysis")

if __name__ == "__main__":
    main()