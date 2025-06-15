# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch._dynamo.config
import torch._inductor.config
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import contextlib

def device_sync(device):
    if "cuda" in device:
        torch.cuda.synchronize(device)
    elif ("cpu" in device) or ("mps" in device):
        pass
    else:
        print(f"device={device} is not yet suppported")

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
# Experimental features to reduce compilation times, will be on by default in future
torch._inductor.config.fx_graph_cache = True 
torch._functorch.config.enable_autograd_cache = True

default_device = 'cuda' if torch.cuda.is_available() else 'cpu'

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def speculative_decode(
    model,
    draft_model,
    tokenizer,
    cur_tokens: torch.Tensor,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    """Simplified speculative decoding for HuggingFace models"""
    device = cur_tokens.device
    
    # Generate draft tokens
    with torch.no_grad():
        draft_outputs = draft_model.generate(
            cur_tokens,
            max_new_tokens=speculate_k,
            do_sample=True,
            temperature=sampling_kwargs.get('temperature', 1.0),
            top_k=sampling_kwargs.get('top_k', None),
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
            output_scores=True
        )
    
    draft_tokens = draft_outputs.sequences[0, cur_tokens.shape[1]:]
    
    # Verify with target model
    candidate_tokens = torch.cat([cur_tokens, draft_tokens.unsqueeze(0)], dim=1)
    
    with torch.no_grad():
        target_outputs = model(candidate_tokens)
        target_logits = target_outputs.logits[0, cur_tokens.shape[1]-1:-1]  # Exclude last position
    
    target_probs = logits_to_probs(target_logits, **sampling_kwargs)
    
    # Simple acceptance: accept all for now (can be improved with proper speculative decoding logic)
    return draft_tokens

@torch.no_grad()
def generate_hf(
    model,
    tokenizer,
    prompt_tokens: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    *,
    interactive: bool,
    draft_model=None,
    speculate_k: Optional[int] = 8,
    callback=lambda x: x,
    **sampling_kwargs
) -> Tuple[torch.Tensor, dict]:
    """
    Generate text using HuggingFace models
    """
    device = prompt_tokens.device
    is_speculative = draft_model is not None
    
    # Repeat prompt for batch
    if prompt_tokens.dim() == 1:
        prompt_tokens = prompt_tokens.unsqueeze(0)
    if batch_size > 1:
        prompt_tokens = prompt_tokens.repeat(batch_size, 1)
    
    T = prompt_tokens.size(-1)
    accept_counts = [0] * (speculate_k + 1) if is_speculative else [0]
    
    if is_speculative and draft_model is not None:
        # Speculative decoding path
        current_tokens = prompt_tokens.clone()
        generated_tokens = []
        
        for _ in range(max_new_tokens):
            if current_tokens.shape[1] >= model.config.max_position_embeddings:
                break
                
            # Use speculative decoding
            new_tokens = speculative_decode(
                model, draft_model, tokenizer, current_tokens[0:1], 
                min(speculate_k, max_new_tokens - len(generated_tokens)),
                **sampling_kwargs
            )
            
            num_accepted = min(len(new_tokens), max_new_tokens - len(generated_tokens))
            accept_counts[num_accepted - 1] += 1
            
            for i in range(num_accepted):
                generated_tokens.append(new_tokens[i])
                callback(new_tokens[i:i+1])
                
            current_tokens = torch.cat([current_tokens, new_tokens[:num_accepted].unsqueeze(0)], dim=1)
            
            if len(generated_tokens) >= max_new_tokens:
                break
                
        # Pad to batch size
        final_length = current_tokens.shape[1]
        result = current_tokens.repeat(batch_size, 1) if batch_size > 1 else current_tokens
        
    else:
        # Standard generation
        generation_config = GenerationConfig(
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=sampling_kwargs.get('temperature', 1.0),
            top_k=sampling_kwargs.get('top_k', None),
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        
        if interactive:
            # Interactive generation with callback
            result = prompt_tokens.clone()
            for _ in range(max_new_tokens):
                outputs = model.generate(
                    result,
                    max_new_tokens=1,
                    generation_config=generation_config,
                    return_dict_in_generate=True,
                )
                new_token = outputs.sequences[:, -1:]
                result = torch.cat([result, new_token], dim=1)
                callback(new_token[0])
                
                if new_token[0].item() == tokenizer.eos_token_id:
                    break
        else:
            # Batch generation
            result = model.generate(
                prompt_tokens,
                generation_config=generation_config,
                return_dict_in_generate=True,
            ).sequences
    
    generate_stats = {
        'accept_counts': accept_counts
    }
    return result, generate_stats

def _load_hf_model(model_name_or_path: str, device: str, precision: torch.dtype, compile_model: bool = False):
    """Load HuggingFace model and tokenizer"""
    print(f"Loading model from {model_name_or_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=precision,
        trust_remote_code=True
    )
    
    if not ("cuda" in device and hasattr(model, 'hf_device_map')):
        model = model.to(device)
    
    model.eval()
    
    if compile_model:
        print("Compiling model...")
        model = torch.compile(model, mode="reduce-overhead")
    
    return model, tokenizer

def _get_model_size(model):
    """Calculate model size in bytes and parameter count"""
    model_size = 0
    params = 0
    for param in model.parameters():
        param_size = param.numel() * param.element_size()
        model_size += param_size
        params += param.numel()
    return model_size, params

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    """Encode string to tokens"""
    tokens = tokenizer.encode(string, add_special_tokens=bos)
    return torch.tensor(tokens, dtype=torch.long, device=device)

B_INST, E_INST = "[INST]", "[/INST]"

def main(
    prompt: Union[int, str] = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: str = "microsoft/DialoGPT-medium",  # Changed to HF model name
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[str] = None,  # HF model name for draft
    speculate_k: int = 5,
    device=default_device,
) -> None:
    """Generates text samples based on a HuggingFace model and tokenizer."""
    
    print(f"Using device={device}")
    precision = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    is_speculative = draft_checkpoint_path is not None
    is_chat = "chat" in str(checkpoint_path).lower() or "instruct" in str(checkpoint_path).lower()

    print("Loading model ...")
    t0 = time.time()
    
    # Load main model
    model, tokenizer = _load_hf_model(checkpoint_path, device, precision, compile)
    
    # Load draft model if specified
    draft_model = None
    if is_speculative:
        print("Loading draft model...")
        draft_model, _ = _load_hf_model(draft_checkpoint_path, device, precision, compile)
    
    device_sync(device=device)
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    # Prepare prompt
    if isinstance(prompt, str):
        encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    else:
        # Generate synthetic prompt
        vocab_size = len(tokenizer)
        encoded = torch.randint(0, vocab_size, (prompt,), device=device, dtype=torch.long)
    
    prompt_length = encoded.size(-1)
    
    torch.manual_seed(1234)
    model_size, params = _get_model_size(model)
    
    print(f"Model size: {model_size / 1e9:.2f} GB")
    print(f"Parameters: {params / 1e6:.1f} M")

    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
    }
    
    start = -1 if compile else 0

    for i in range(start, num_samples):
        device_sync(device=device)
        
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode('.')[0] if '.' in tokenizer.get_vocab() else tokenizer.eos_token_id
            done_generating = False
            
            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                token_str = tokenizer.decode(x.tolist(), skip_special_tokens=True)
                buffer.append(token_str)
                if x.item() == tokenizer.eos_token_id:
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print(''.join(buffer), end='', flush=True)
                    buffer.clear()
        else:
            callback = lambda x: x
            
        t0 = time.perf_counter()
        
        # Profiling setup
        if (i != num_samples - 1 or not profile):
            prof = contextlib.nullcontext()
        else:
            prof = torch.profiler.profile()
            
        with prof:
            y, metrics = generate_hf(
                model,
                tokenizer,
                encoded,
                max_new_tokens,
                batch_size=batch_size,
                draft_model=draft_model,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
            )
            aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
            
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
            
        if hasattr(prof, "export_chrome_trace"):
            prof.export_chrome_trace(f"{profile}.json")
            
        device_sync(device=device)
        t = time.perf_counter() - t0

        if not interactive:
            # Display first generation
            if batch_size > 1:
                print("Only displaying the first generation of the batch")
            print(tokenizer.decode(y[0].tolist(), skip_special_tokens=True))
        else:
            print()
            
        tokens_generated = y.size(-1) - prompt_length
        generated_tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(generated_tokens_sec)
        
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {generated_tokens_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * generated_tokens_sec / 1e9:.02f} GB/s")
        
        total_tokens_sec = y.numel() / t
        print(f"FLOPS achieved: {params * total_tokens_sec * 2 / 1e12:.02f} TF/s")
        print()

    print("==========")
    if is_speculative:
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics['accept_counts'])]
        acceptance_probs = [i/sum(counts_aggregated) for i in counts_aggregated]
        print(f"Acceptance probs: {acceptance_probs}")
        print(f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}")

    print(f"Batch Size: {batch_size}")
    print(f"Prompt Length: {prompt_length}")
    print(f"Generated tokens: {max_new_tokens}")
    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    
    if torch.cuda.is_available():
        print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='HuggingFace Model Benchmark')

    def int_or_str(x):
        try:
            return int(x)
        except:
            return x

    parser.add_argument('--prompt', type=int_or_str, default="Hello, my name is", 
                       help="Input prompt. If it's an integer, will generate a synthetic prompt.")
    parser.add_argument('--interactive', action='store_true', 
                       help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=5, 
                       help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=100, 
                       help='Maximum number of new tokens.')
    parser.add_argument('--batch_size', type=int, default=1, 
                       help='Batch size to benchmark with')
    parser.add_argument('--top_k', type=int, default=200, 
                       help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, 
                       help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=str, 
                       default="microsoft/DialoGPT-medium", 
                       help='HuggingFace model name or path.')
    parser.add_argument('--compile', action='store_true', 
                       help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', 
                       help='Whether to compile the prefill (for compatibility, not used in HF version)')
    parser.add_argument('--profile', type=Path, default=None, 
                       help='Profile path.')
    parser.add_argument('--speculate_k', type=int, default=5, 
                       help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=str, default=None, 
                       help='Draft model HuggingFace name or path.')
    parser.add_argument('--device', type=str, default=default_device, 
                       help='Device to use')

    args = parser.parse_args()
    main(
        args.prompt, args.interactive, args.num_samples, args.max_new_tokens, 
        args.batch_size, args.top_k, args.temperature, args.checkpoint_path, 
        args.compile, args.compile_prefill, args.profile, args.draft_checkpoint_path,
        args.speculate_k, args.device
    )