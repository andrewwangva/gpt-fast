#!/usr/bin/env python3
"""
Simple Speculative Decoding Diagnostics Tool - Fixed Device Issues

A diagnostic tool that follows the exact pattern of the original generate.py
to avoid device mismatch issues.
"""

import sys
import argparse
import torch
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import json

# Import from the existing generate.py
from generate import (
    Transformer, get_tokenizer, encode_tokens, _load_model, 
    logits_to_probs, multinomial_sample_one_no_sync, model_forward,
    decode_n_tokens, prefill, default_device, pad_vocab, generate,
    speculative_decode
)

def entropy(probs: torch.Tensor) -> torch.Tensor:
    """Calculate entropy of probability distribution"""
    eps = 1e-10
    return -torch.sum(probs * torch.log(probs + eps), dim=-1)

def diagnostic_speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    tokenizer,
    step_num: int,
    prefix_tokens: Optional[torch.Tensor] = None,
    **sampling_kwargs
) -> Tuple[torch.Tensor, Dict]:
    """
    A diagnostic version of speculative_decode that provides detailed analysis
    """
    print(f"\n{'='*80}")
    print(f"🔍 DIAGNOSTIC STEP {step_num} - Position {input_pos}")
    print(f"{'='*80}")
    if prefix_tokens is not None:
        # Work on CPU for decoding; truncate to last N chars to avoid spam
        prefix_ids = prefix_tokens.detach().cpu().tolist()
        prefix_text = tokenizer.decode(prefix_ids)
        # Optional: only show last X chars/tokens for readability
        print(f"🧩 PREFIX (len={len(prefix_ids)} tokens)")
        print(prefix_text[-600:])   # show the tail; adjust 600 → taste
        print("-" * 80)
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=device)
    
    # Step 1: Draft model generates tokens
    print(f"📝 Draft model generating {speculate_k} tokens...")
    draft_start = time.time()
    
    draft_tokens, draft_probs = decode_n_tokens(
        draft_model, cur_token.view(1, -1), orig_input_pos.clone(), 
        speculate_k, **sampling_kwargs
    )
    draft_tokens = torch.cat(draft_tokens)  # [L, 1]
    
    draft_time = time.time() - draft_start
    print(f"   ⏱️  Draft generation: {draft_time:.4f}s")
    
    # Step 2: Target model verification
    print(f"🎯 Target model verifying {len(draft_tokens)} tokens...")
    target_start = time.time()
    
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1, 1), draft_tokens], dim=0).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=device)
    )
    target_probs = logits_to_probs(target_logits, **sampling_kwargs)
    draft_probs_tensor = torch.cat(draft_probs, dim=0).unsqueeze(0)  # [B, L, vocab_size]
    
    target_time = time.time() - target_start
    print(f"   ⏱️  Target verification: {target_time:.4f}s")
    
    # Step 3: Analyze each token
    print(f"\n📊 TOKEN ANALYSIS:")
    print(f"{'Pos':<3} {'Token':<15} {'Draft_P':<8} {'Target_P':<8} {'Accept_P':<8} {'Random':<8} {'Decision':<10} {'Entropies':<15}")
    print(f"{'-'*85}")
    
    diagnostics = {
        'step': step_num,
        'input_pos': input_pos,
        'timing': {'draft': draft_time, 'target': target_time},
        'tokens': [],
        'acceptance_info': {},
        'outcome': {}
    }
    
    # Follow the exact logic from the original speculative_decode
    p = draft_probs_tensor[:, torch.arange(0, speculate_k, device=device), draft_tokens.view(-1)]  # [B, L]
    q = target_probs[:, torch.arange(0, speculate_k, device=device), draft_tokens.view(-1)]  # [B, L]
    
    # Create verification result (simplified version of verify_tokens for diagnostics)
    accept_draft_prob = torch.minimum(torch.ones_like(q), q / (p + 1e-10))
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()
    
    # Analyze each token
    for i in range(speculate_k):
        token_id = draft_tokens[i].item()
        token_str = tokenizer.decode([token_id])
        
        draft_prob = p[0, i].item()
        target_prob = q[0, i].item()
        accept_prob = accept_draft_prob[0, i].item()
        
        # Calculate entropies
        draft_dist = draft_probs_tensor[0, i]
        target_dist = target_probs[0, i]
        draft_entropy = entropy(draft_dist).item()
        target_entropy = entropy(target_dist).item()
        
        # Determine if this token was accepted
        was_rejected = len(rejected_locations) > 0 and rejected_locations[0, 1].item() <= i
        decision = "❌ REJECT" if was_rejected else "✅ ACCEPT"
        
        print(f"{i:<3} '{token_str}':<13 {draft_prob:<8.4f} {target_prob:<8.4f} {accept_prob:<8.4f} {'N/A':<8} {decision:<10} D:{draft_entropy:.1f}/T:{target_entropy:.1f}")
        
        diagnostics['tokens'].append({
            'position': i,
            'token_id': token_id,
            'token_str': token_str,
            'draft_prob': draft_prob,
            'target_prob': target_prob,
            'acceptance_prob': accept_prob,
            'draft_entropy': draft_entropy,
            'target_entropy': target_entropy,
            'accepted': not was_rejected
        })
    
    # Apply the original logic for final result
    if len(rejected_locations) == 0:  # All draft tokens accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[:, -1])
        
        # Keep draft model in sync
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        
        result = torch.cat([draft_tokens.view(-1), last_token[0].view(-1)], dim=0)
        
        print(f"\n🎉 ALL TOKENS ACCEPTED! Final token: '{tokenizer.decode([last_token[0].item()])}'")
        
        diagnostics['outcome'] = {
            'type': 'all_accepted',
            'tokens_used': len(result),
            'final_token': tokenizer.decode([last_token[0].item()])
        }
        
    else:
        accept_length = rejected_locations[0, 1].item()
        p_rejected = draft_probs_tensor[:, accept_length]
        q_rejected = target_probs[:, accept_length]
        new_dist = q_rejected - p_rejected
        new_dist = torch.where(new_dist > 0, new_dist, 0.0)
        new_dist = new_dist / new_dist.sum()
        next_token = multinomial_sample_one_no_sync(new_dist)
        
        result = torch.cat([draft_tokens[:accept_length].view(-1), next_token.view(-1)], dim=0)
        
        original_token = tokenizer.decode([draft_tokens[accept_length].item()])
        corrected_token = tokenizer.decode([next_token.item()])
        
        print(f"\n❌ REJECTION at position {accept_length}")
        print(f"🔄 Correction: '{original_token}' → '{corrected_token}'")
        
        diagnostics['outcome'] = {
            'type': 'rejected',
            'rejection_pos': accept_length,
            'tokens_used': len(result),
            'original_token': original_token,
            'corrected_token': corrected_token
        }
    
    # Summary statistics
    accepted_count = len([t for t in diagnostics['tokens'] if t['accepted']])
    acceptance_rate = accepted_count / len(diagnostics['tokens'])
    avg_draft_entropy = sum(t['draft_entropy'] for t in diagnostics['tokens']) / len(diagnostics['tokens'])
    avg_target_entropy = sum(t['target_entropy'] for t in diagnostics['tokens']) / len(diagnostics['tokens'])
    
    diagnostics['acceptance_info'] = {
        'drafted': len(diagnostics['tokens']),
        'accepted': accepted_count,
        'acceptance_rate': acceptance_rate,
        'avg_draft_entropy': avg_draft_entropy,
        'avg_target_entropy': avg_target_entropy,
        'tokens_used': diagnostics['outcome']['tokens_used']
    }
    
    print(f"\n📊 SUMMARY:")
    print(f"   📝 Drafted: {len(diagnostics['tokens'])}")
    print(f"   ✅ Accepted: {accepted_count}")
    print(f"   📈 Acceptance rate: {acceptance_rate:.1%}")
    print(f"   ⚡ Tokens used: {diagnostics['outcome']['tokens_used']}")
    print(f"   🧠 Avg entropies - Draft: {avg_draft_entropy:.2f}, Target: {avg_target_entropy:.2f}")
    
    return result, diagnostics

def monkey_patch_speculative_decode(tokenizer, initial_prefix: torch.Tensor):
    """Replace the original speculative_decode with our diagnostic version, tracking prefix."""
    import generate
    
    step_counter = [0]              # mutable counter
    all_diagnostics = []
    prefix_tokens = initial_prefix.detach().cpu().tolist()  # running prefix (ids)

    original_speculative_decode = generate.speculative_decode

    def diagnostic_wrapper(model, draft_model, cur_token, input_pos, speculate_k, **kwargs):
        # Build the prefix up to *and including* the current token
        current_prefix = torch.tensor(
            prefix_tokens + [int(cur_token.item())],
            dtype=torch.long,
            device=cur_token.device
        )

        result, diag = diagnostic_speculative_decode(
            model, draft_model, cur_token, input_pos, speculate_k,
            tokenizer, step_counter[0],
            prefix_tokens=current_prefix, 
            **kwargs
        )
        step_counter[0] += 1
        all_diagnostics.append(diag)

        # Update running prefix with tokens produced *this* step
        prefix_tokens.extend(result.detach().cpu().tolist())
        return result
    
    generate.speculative_decode = diagnostic_wrapper
    return all_diagnostics

def main():
    parser = argparse.ArgumentParser(description='Simple Speculative Decoding Diagnostics')
    parser.add_argument('--checkpoint_path', type=Path, required=True)
    parser.add_argument('--draft_checkpoint_path', type=Path, required=True)
    parser.add_argument('--prompt', type=str, default="The quick brown fox")
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--speculate_k', type=int, default=5)
    parser.add_argument('--temperature', type=float, default=0.8)
    parser.add_argument('--top_k', type=int, default=200)
    parser.add_argument('--device', type=str, default=default_device)
    parser.add_argument('--save_diagnostics', type=str, help='Save diagnostics to JSON file')
    
    args = parser.parse_args()
    
    print("🔧 Loading models...")
    
    # Load models exactly like the original
    precision = torch.bfloat16
    model = _load_model(args.checkpoint_path, args.device, precision, use_tp=False)
    draft_model = _load_model(args.draft_checkpoint_path, args.device, precision, use_tp=False)
    
    # Pad vocab if needed
    if draft_model.output.weight.shape[0] != model.output.weight.shape[0]:
        pad_vocab(draft_model, model.output.weight.shape[0])
    
    # Load tokenizer
    tokenizer_path = args.checkpoint_path.parent / "tokenizer.json"
    tokenizer = get_tokenizer(tokenizer_path, args.checkpoint_path)
    
    # Encode prompt
    encoded = encode_tokens(tokenizer, args.prompt, bos=True, device=args.device)
    
    print(f"✅ Models loaded!")
    print(f"📝 Prompt: '{args.prompt}'")
    print(f"🔢 Encoded length: {encoded.size(-1)} tokens")
    
    # Monkey patch the speculative decode function to add diagnostics
    all_diagnostics = monkey_patch_speculative_decode(tokenizer, encoded)
    
    print(f"\n🚀 STARTING GENERATION WITH DIAGNOSTICS...")
    
    # Use the original generate function with our diagnostic wrapper
    seq, generate_stats = generate(
        model=model,
        prompt=encoded,
        max_new_tokens=args.max_new_tokens,
        batch_size=1,
        interactive=False,
        draft_model=draft_model,
        speculate_k=args.speculate_k,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    
    # Display results
    print(f"\n{'='*80}")
    print(f"📜 GENERATED TEXT:")
    print(f"{'='*80}")
    generated_text = tokenizer.decode(seq[0].tolist())
    print(generated_text)
    print(f"{'='*80}")
    
    # Overall statistics
    if all_diagnostics:
        total_drafted = sum(d['acceptance_info']['drafted'] for d in all_diagnostics)
        total_accepted = sum(d['acceptance_info']['accepted'] for d in all_diagnostics)
        overall_acceptance = total_accepted / total_drafted if total_drafted > 0 else 0
        
        print(f"\n🏁 OVERALL STATISTICS:")
        print(f"   🔢 Total steps: {len(all_diagnostics)}")
        print(f"   📝 Total tokens drafted: {total_drafted}")
        print(f"   ✅ Total tokens accepted: {total_accepted}")
        print(f"   📊 Overall acceptance rate: {overall_acceptance:.1%}")
        print(f"   ⚡ Average tokens per step: {total_accepted / len(all_diagnostics):.2f}")
    
    # Save diagnostics if requested
    if args.save_diagnostics:
        with open(args.save_diagnostics, 'w') as f:
            json.dump(all_diagnostics, f, indent=2, default=lambda x: float(x) if torch.is_tensor(x) else x)
        print(f"💾 Diagnostics saved to {args.save_diagnostics}")

if __name__ == "__main__":
    main()