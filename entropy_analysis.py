# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import json
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch._dynamo.config
import torch._inductor.config
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from torch.nn import functional as F


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

create_block_mask = torch.compile(create_block_mask)

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from model import Transformer
from tokenizer import get_tokenizer

def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    
    if top_p is not None and top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        probs = torch.nn.functional.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(probs, dim=-1)

        sorted_mask = cumulative_probs <= top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = True 

        masked_logits = sorted_logits.masked_fill(~sorted_mask, -float("Inf"))
        logits = torch.full_like(logits, -float("Inf"))
        logits.scatter_(-1, sorted_indices, masked_logits)

    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None, top_p: Optional[float] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k, top_p)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def roundup(val, multiplier):
    return ((val - 1) // multiplier + 1) * multiplier

def causal_mask(b, h, q, kv):
    return q >= kv

def prefill(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    #mask = create_block_mask(causal_mask, 1, 1, input_pos.shape[0], model.max_seq_length, device=x.device)
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: Transformer, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    #block_index = input_pos // block_mask.BLOCK_SIZE[0]
    #mask = block_mask[:, :, block_index]
    #mask.mask_mod = block_mask.mask_mod
    #mask.seq_lengths = (1, model.max_seq_length)
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: Transformer, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, eos_id=None, **sampling_kwargs):
    #block_mask = create_block_mask(causal_mask, 1, 1, model.max_seq_length, model.max_seq_length, device=cur_token.device)
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        next_token, next_prob = decode_one_token(
            model, cur_token, input_pos, **sampling_kwargs
        )
        token_id = next_token.item()
        input_pos += 1
        new_tokens.append(next_token.clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        cur_token = next_token.clone()

        if eos_id is not None and token_id == eos_id:
            break
    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)


def verify_tokens(p, q):
    """
    p has shape [B, spec, 1] and represents draft tokens
    q has shape [B, L, 1] and represents draft tokens
    """
    return (q < 0.1).nonzero()
def speculative_decode(
    model: Transformer,
    draft_model: Transformer,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)
    draft_tokens = torch.cat(draft_tokens) # [L, 1]
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1, 1), draft_tokens], dim=0).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    target_probs = logits_to_probs(target_logits, **sampling_kwargs)
    draft_probs = torch.cat(draft_probs, dim=0).unsqueeze(0) #[B, L, 1]

    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token

    
    p = draft_probs[:, torch.arange(0, speculate_k, device=device), draft_tokens.view(-1)] # [B, L, 1]
    q = target_probs[:, torch.arange(0, speculate_k, device=device), draft_tokens.view(-1)] # [B, L, 1]

    rejected_locations = verify_tokens(p, q)
    #accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    #rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero() #
    if len(rejected_locations) == 0: # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[:, -1])

        return torch.cat([draft_tokens.view(-1), last_token[0].view(-1)], dim=0)
    else:
        accept_length = rejected_locations[0, 1].item() # [B]
        """
        p = draft_probs[:, accept_length]
        q = target_probs[:, accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        """
        last_token = multinomial_sample_one_no_sync(target_probs[:, accept_length])
        return torch.cat([draft_tokens[:accept_length].view(-1), last_token.view(-1)], dim=0)

def pad_vocab(model: Transformer, target_vocab_size: int):
    cur_vocab_size = model.output.weight.shape[0]
    if cur_vocab_size >= target_vocab_size:
        return model  # nothing to do

    print(f"Patching vocab: {cur_vocab_size} -> {target_vocab_size}")
    device = model.output.weight.device
    dtype = model.output.weight.dtype

    # Pad output projection
    padding = torch.zeros((target_vocab_size - cur_vocab_size, model.output.weight.shape[1]), device=device, dtype=dtype)
    model.output.weight = torch.nn.Parameter(torch.cat([model.output.weight.data, padding], dim=0))

    # Pad token embedding
    tok_emb_padding = torch.zeros((target_vocab_size - cur_vocab_size, model.tok_embeddings.weight.shape[1]), device=device, dtype=dtype)
    model.tok_embeddings.weight = torch.nn.Parameter(torch.cat([model.tok_embeddings.weight.data, tok_emb_padding], dim=0))

    return model

def compute_entropies(probabilities):
    # Handle the case where probabilities are 0 to avoid NaN from log(0)
    # Use torch.where to set p*log(p) = 0 when p = 0 (limit as p->0)
    log_probs = torch.log(probabilities + 1e-12)  # Add small epsilon to avoid log(0)
    return -torch.sum(probabilities * log_probs, dim=-1)

def process_batch(
    problem: str,
    model: "Transformer",
    tokenizer,
    device,
    max_new_tokens: int,
    draft_model: Optional["Transformer"],
    speculate_k: int,
    temperature: float,
    top_k: int,
    eos_id: Optional[int],
    batch_size: int = 1,
    expected_answer: str = None,
    callback=lambda x: x,
):
    """
    Process a single prompt/problem and return a dict with results.
    """
    prompt = tokenizer.render_chat([
        {"role": "system", "content": ""},
        {"role": "user", "content": problem}
    ])

    encoded = encode_tokens(tokenizer, prompt, bos=False, device=device)

    torch.manual_seed(1234)  # for reproducibility

    t0 = time.perf_counter()
    y, generate_stats = generate(
        model,
        encoded,
        max_new_tokens,
        batch_size=batch_size,
        draft_model=draft_model,
        speculate_k=speculate_k,
        interactive=False,
        callback=callback,
        temperature=temperature,
        top_k=top_k,
        eos_id=eos_id,
    )
    t = time.perf_counter() - t0
    prompt_length = encoded.size(-1)
    tokens_generated = y.size(-1) - prompt_length
    print(f"Average tokens/sec: {tokens_generated / t:.2f}")
    output_text = tokenizer.decode(y[0].tolist(), skip_special_tokens=False)
    # Store only top-k probabilities to avoid memory issues
    top_k_probs = []
    for prob in generate_stats['generated_probs']:
        top_probs, top_indices = torch.topk(prob, k=min(10, prob.size(-1)))
        top_k_probs.append({
            "values": top_probs.tolist(),
            "indices": top_indices.tolist()
        })
    
    return {
        "problem": problem,
        "top_probabilities": top_k_probs,  # Much smaller: only top-5 per token
        "entropies": compute_entropies(torch.cat(generate_stats['generated_probs'], dim=0)).tolist(),
        "encoded_tokens": encoded.squeeze().tolist(),  # Convert tensor to list
        "generated_tokens": [token.item() for token in generate_stats['generated_tokens']],  # Convert list of tensors to list of ints
        "expected_answer": expected_answer,
        "llm_output": output_text
    }



@torch.no_grad()
def generate(
    model: Transformer,
    prompt: torch.Tensor,
    max_new_tokens: int,
    batch_size: int,
    *,
    interactive: bool,
    draft_model: Transformer,
    speculate_k: Optional[int] = 8,
    callback = lambda x: x,
    eos_id: Optional[int] = None,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(-1)
    T_new = T + max_new_tokens
    if interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.max_position_embeddings)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length

    generation_limit = max_seq_length - 50
    with torch.device(device):
        print("setup", batch_size, max_seq_length)
        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty(batch_size, T_new, dtype=dtype, device=device)
    # We are just making the same prompt for every batch
    prompt = prompt.view(1, -1).repeat(batch_size, 1)
    empty[:, :T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)
    print(f"prefill x shape: {prompt.view(batch_size, -1).view(batch_size, -1).shape}, input_pos: {input_pos.shape}")
    print("devices", prompt.device, input_pos.device)
    next_token = prefill(model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs).clone()
    if is_speculative:
        prefill(draft_model, prompt.view(batch_size, -1), input_pos, **sampling_kwargs)
    seq[:, T] = next_token.squeeze()

    input_pos = torch.tensor([T], device=device, dtype=torch.int)
    accept_counts = [0] * (speculate_k + 1)

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )
            
            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[:, input_pos + 1 : input_pos + num_added + 1] = next_tokens[:num_added]
            for i in next_tokens[:num_added]:
                callback(i)
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:
        print(f"x shape: {next_token.view(batch_size, -1).shape}, input_pos: {input_pos.shape}")
        remaining_tokens = min(max_new_tokens - 1, generation_limit - T - 1)
        if remaining_tokens <= 0:
            remaining_tokens = 1
        generated_tokens, generated_probs = decode_n_tokens(model, next_token.view(batch_size, -1), input_pos, remaining_tokens, callback=callback, eos_id=eos_id, **sampling_kwargs)
        print(generated_tokens[0].shape, len(generated_tokens))
        generated = torch.cat(generated_tokens, dim=-1)
        print("generated shape:", generated.shape)
        print("seq shape:", seq.shape)
        seq[:, T + 1:T + 1 + generated.size(-1)] = generated
        seq = seq[:, :T + 1 + generated.size(-1)]  # trim to the actual size

    generate_stats = {
        'accept_counts': accept_counts,
        'generated_tokens': generated_tokens,
        'generated_probs': generated_probs,
    }
    return seq, generate_stats

def encode_tokens(tokenizer, string, bos=True, device=default_device):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def encode_batch(
        tokenizer,
        prompts: list[str],                # raw user prompts
        bos: bool = True,
        device: str = default_device
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns
    -------
    prompt_ids : LongTensor [B, T_max]  – left-padded with PAD (or EOS) tokens
    seq_lens   : LongTensor [B]         – original lengths, incl. optional <BOS>
    """
    PAD = tokenizer.eos_id()            # nothing special in model about PAD vs EOS
    seqs = [tokenizer.encode(p) for p in prompts]
    if bos:
        seqs = [[tokenizer.bos_id(), *s] for s in seqs]

    lens = torch.tensor([len(s) for s in seqs], device=device)
    t_max = int(lens.max())
    batch = torch.full((len(seqs), t_max), PAD,
                       dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        batch[i, : len(s)] = torch.tensor(s, device=device)

    return batch, lens


def _load_model(checkpoint_path, device, precision, use_tp):
    use_cuda = 'cuda' in device
    with torch.device('meta'):
        print("checkpoint_path.parent.name", checkpoint_path.parent.name)
        model = Transformer.from_name(checkpoint_path.parent.name)

    if "int8" in str(checkpoint_path):
        print("Using int8 weight-only quantization!")
        from quantize import WeightOnlyInt8QuantHandler
        simple_quantizer = WeightOnlyInt8QuantHandler(model)
        model = simple_quantizer.convert_for_runtime()

    if "int4" in str(checkpoint_path):
        print("Using int4 weight-only quantization!")
        path_comps = checkpoint_path.name.split(".")
        groupsize = int(path_comps[-2][1:])
        from quantize import WeightOnlyInt4QuantHandler
        simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
        model = simple_quantizer.convert_for_runtime()

    checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
    if "model" in checkpoint and "stories" in str(checkpoint_path):
        checkpoint = checkpoint["model"]
    model.load_state_dict(checkpoint, assign=True)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()

def _get_model_size(model):
    model_size = 0
    params = 0
    for name, child in model.named_children():
        if not isinstance(child, torch.nn.Embedding):
            model_size += sum(
                [
                    p.numel() * p.dtype.itemsize
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
            params += sum(
                [
                    p.numel()
                    for p in itertools.chain(child.parameters(), child.buffers())
                ]
            )
    return model_size, params

def load_jsonl(path):
    with open(path, "r") as f:
        for line in f:
            yield json.loads(line)

B_INST, E_INST = "[INST]", "[/INST]"

def print_gpu_memory_usage(stage=""):
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB
        reserved = torch.cuda.memory_reserved() / 1024**3    # Convert to GB
        max_allocated = torch.cuda.max_memory_allocated() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_memory = total_memory - reserved
        
        print(f"=== GPU Memory {stage} ===")
        print(f"Allocated: {allocated:.2f} GB")
        print(f"Reserved: {reserved:.2f} GB") 
        print(f"Max Allocated: {max_allocated:.2f} GB")
        print(f"Total GPU Memory: {total_memory:.2f} GB")
        print(f"Free Memory: {free_memory:.2f} GB")
        print(f"Memory Usage: {(reserved/total_memory)*100:.1f}%")
        print("=" * 40)

def clear_caches(model):
    """Clear KV caches to free memory"""
    """Comprehensive cache clearing to free memory"""
    # Clear layer-specific caches
    for layer in model.layers:
        if hasattr(layer, 'attention'):
            if hasattr(layer.attention, 'kv_cache') and layer.attention.kv_cache is not None:
                del layer.attention.kv_cache
                layer.attention.kv_cache = None
            if hasattr(layer.attention, 'k_cache') and layer.attention.k_cache is not None:
                del layer.attention.k_cache
                layer.attention.k_cache = None
            if hasattr(layer.attention, 'v_cache') and layer.attention.v_cache is not None:
                del layer.attention.v_cache
                layer.attention.v_cache = None
    
    # Clear model-level cached tensors
    attrs_to_clear = ['cos', 'sin', 'causal_mask', 'freqs_cis', 'mask_cache']
    for attr in attrs_to_clear:
        if hasattr(model, attr) and getattr(model, attr) is not None:
            delattr(model, attr)
    
    # Reset cache size tracking
    model.max_seq_length = -1
    model.max_batch_size = -1
    
    # Force garbage collection and clear CUDA cache
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()


def compute_score(solution_str, ground_truth) -> float:
    retval = 0.0
    try:
        string_in_last_boxed = last_boxed_only_string(solution_str)
        if string_in_last_boxed is not None:
            answer = remove_boxed(string_in_last_boxed)
            if is_equiv(answer, ground_truth):
                retval = 1.0
    except Exception as e:
        print(e)

    return retval


# string normalization from https://github.com/EleutherAI/lm-evaluation-harness/blob/master/lm_eval/tasks/hendrycks_math.py
def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


def remove_boxed(s):
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]

    left = "\\boxed{"

    assert s[: len(left)] == left
    assert s[-1] == "}"

    return s[len(left) : -1]


def last_boxed_only_string(string):
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    retval = None if right_brace_idx is None else string[idx : right_brace_idx + 1]

    return retval


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:  # noqa: E722
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except:  # noqa: E722
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace("\n", "")

    # remove inverse spaces
    string = string.replace("\\!", "")

    # replace \\ with \
    string = string.replace("\\\\", "\\")

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")

    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")  # noqa: W605

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")

    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1).
    # Also does a/b --> \\frac{a}{b}
    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"

    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = fix_a_slash_b(string)

    return string

def main(
    dataset_path: Optional[Path] = None,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    batch_size: int = 1,
    top_k: int = 200,
    top_p: Optional[int] = None,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    compile: bool = True,
    compile_prefill: bool = False,
    draft_checkpoint_path: Optional[Path] = None,
    speculate_k: int = 5,
    generate_initial_rollouts: bool = True,
    device=default_device,
) -> None:

    global print

    assert checkpoint_path.is_file(), checkpoint_path

    tokenizer_path = checkpoint_path.parent / "tokenizer.json"
    assert tokenizer_path.is_file(), str(tokenizer_path)
    
    from tp import maybe_init_dist
    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    print(f"Using device={device}")
    precision = torch.bfloat16
    is_speculative = draft_checkpoint_path is not None

    print("Loading model ...")
    t0 = time.time()
    model = _load_model(checkpoint_path, device, precision, use_tp)

    if is_speculative:
        draft_model = _load_model(draft_checkpoint_path, device, precision, use_tp)
        if draft_model.output.weight.shape[0] != model.output.weight.shape[0]:
            pad_vocab(draft_model, model.output.weight.shape[0])
    else:
        draft_model = None

    device_sync(device=device) # MKG
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    torch.manual_seed(1234)
    model_size, params = _get_model_size(model)
    if compile:
        if is_speculative and use_tp: # and ("cuda" in device):
            torch._inductor.config.triton.cudagraph_trees = False # Bug with cudagraph trees in this case

        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        # Uncomment to squeeze more perf out of prefill
        if compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)
    
    if generate_initial_rollouts:
        dataset_path = Path(dataset_path)  # Convert string to Path object
        assert dataset_path.is_file(), f"{dataset_path} not found"
        output_path = dataset_path.parent / f"improved_results_{dataset_path.name}"

        problems = list(load_jsonl(dataset_path))
        tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)
        print(f"Loaded {len(problems)} problems from {dataset_path}")
        results = []

        #compilation run
        if compile:
            prompt = tokenizer.render_chat([
                {"role": "system", "content": ""},
                {"role": "user", "content": problems[0]["problem"]}
            ])

            encoded = encode_tokens(tokenizer, prompt, bos=False, device=device)
            _, _ = generate(
                model,
                encoded,
                max_new_tokens,
                batch_size=batch_size,
                interactive=False,
                draft_model=None,                
            )
        for idx, item in enumerate(problems):
            out = process_batch(
                problem=item["problem"],
                model=model,
                tokenizer=tokenizer,
                device=device,
                max_new_tokens=max_new_tokens,
                draft_model=None,
                speculate_k=speculate_k,
                temperature=temperature,
                top_k=top_k,
                eos_id=tokenizer.eos_id(),
                batch_size=1,
                expected_answer=item.get("answer", None),
            )
            results.append(out)
            print(f"Problem {idx+1}/{len(problems)} done")
        with open(output_path, "w") as f:
            for row in results:
                print(json.dumps(row, ensure_ascii=False), file=f)
        print(f"Saved results to {output_path}")
        return
    else:
        dataset_path = Path(dataset_path)  # Convert string to Path object
        assert dataset_path.is_file(), f"{dataset_path} not found"
        output_path = dataset_path.parent / f"improved_results_{dataset_path.name}"

        results = []
        with open(dataset_path, 'r') as f:
            for line in f:
                results.append(json.loads(line))

        print(f"Loaded {len(results)} problems")

        import numpy as np

        for i, result in enumerate(results):
            print(result.keys())
            entropies = result['entropies']
            
            # Sample positions with high entropy > 0.66
            high_entropy_mask = np.array(entropies) > 0.66
            high_entropy_indices = np.where(high_entropy_mask)[0]
            if len(high_entropy_indices) >= 20:
                sample_high_entropy_positions = np.random.choice(high_entropy_indices, size=20, replace=False)
            else:
                sample_high_entropy_positions = high_entropy_indices

            # Sample positions with low entropy < 0.33
            low_entropy_mask = np.array(entropies) < 0.33
            low_entropy_indices = np.where(low_entropy_mask)[0]
            if len(low_entropy_indices) >= 20:
                sample_low_entropy_positions = np.random.choice(low_entropy_indices, size=20, replace=False)
            else:
                sample_low_entropy_positions = low_entropy_indices

            # Sample positions with medium entropy (0.33 <= entropy <= 0.66)
            medium_entropy_mask = (np.array(entropies) >= 0.33) & (np.array(entropies) <= 0.66)
            medium_entropy_indices = np.where(medium_entropy_mask)[0]
            if len(medium_entropy_indices) >= 20:
                sample_medium_entropy_positions = np.random.choice(medium_entropy_indices, size=20, replace=False)
            else:
                sample_medium_entropy_positions = medium_entropy_indices

            # Process all three entropy categories
            entropy_categories = [
                ("high", sample_high_entropy_positions),
                ("medium", sample_medium_entropy_positions), 
                ("low", sample_low_entropy_positions)
            ]
            
            for entropy_class, sampled_positions in entropy_categories:
                print(f"\nProcessing {entropy_class} entropy positions for problem {i+1}")
                print(f"Number of {entropy_class} entropy positions: {len(sampled_positions)}")
                
                for position in sampled_positions:
                    device_sync(device=device)
                    clear_caches(draft_model)

                    print_gpu_memory_usage("Before clearing caches")
                    max_seq_length = 32050
                    with torch.device(device):
                        print("setup", batch_size, max_seq_length)
                        model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
                        if is_speculative and draft_model is not model:
                            draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
                    
                    tokens = torch.tensor([result['encoded_tokens'] + result['generated_tokens'][:position]], device=device)
                    T = tokens.size(-1)
                    input_pos = torch.arange(0, T, device=device)
                    print(f"Tokens: {tokens.shape}")

                    rollouts = []
                    tokenizer = get_tokenizer(tokenizer_path, checkpoint_path)
                    
                    for rollout_idx in range(8):
                        # Clear caches between rollouts to avoid memory issues
                        clear_caches(model)
                        clear_caches(draft_model)
                        
                        # Setup fresh caches
                        with torch.device(device):
                            model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
                            if is_speculative and draft_model is not model:
                                draft_model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)
                        
                        # Regenerate from this position with different random seed
                        torch.manual_seed(1234 + rollout_idx + position)
                        
                        T = tokens.size(-1)
                        total_length = T + max_new_tokens
                        seq = torch.full((batch_size, total_length), tokenizer.eos_id(), 
                                    dtype=torch.long, device=device)  # Initialize with valid token IDs
                        
                        # Copy input tokens
                        prompt = tokens.view(1, -1).repeat(batch_size, 1)
                        seq[:, :T] = prompt
                        with torch.no_grad():
                            # Generate continuation from the current position
                            rollout_tokens = tokens.clone()
                            T_current = rollout_tokens.size(-1)
                            input_pos_current = torch.arange(0, T_current, device=device)
                            
                            # Prefill up to current position
                            prefill_result = prefill(model, rollout_tokens, input_pos_current, 
                                                temperature=temperature, top_k=top_k)
                            
                            # Generate new tokens from this position
                            input_pos_gen = torch.tensor([T_current], device=device, dtype=torch.int)
                            
                            new_tokens, new_probs = decode_n_tokens(
                                model, 
                                prefill_result.view(batch_size, -1), 
                                input_pos_gen, 
                                max_new_tokens-1,
                                callback=lambda x: x,
                                eos_id=tokenizer.eos_id(),
                                temperature=temperature,
                                top_k=top_k
                            )
                            if new_tokens:
                                concatenated_new_tokens = torch.cat(new_tokens, dim=-1)
                                actual_new_length = concatenated_new_tokens.size(-1)
                                
                                # Ensure we don't exceed sequence bounds
                                max_fill_length = min(actual_new_length, total_length - T - 1)
                                if max_fill_length > 0:
                                    seq[:, T+1:T+1+max_fill_length] = concatenated_new_tokens[:, :max_fill_length]
                                
                                # Trim sequence to actual generated length
                                actual_seq_length = T + 1 + max_fill_length
                                final_seq = seq[:, :actual_seq_length]
                            else:
                                final_seq = seq[:, :T]
                                actual_new_length = 0
                            
                            # Decode to text
                            output_text = tokenizer.decode(seq[0].tolist(), skip_special_tokens=False)
                            
                            # Compute score if we have ground truth
                            score = 0.0
                            if 'expected_answer' in result and result['expected_answer']:
                                score = compute_score(output_text, result['expected_answer'])
                            
                            rollout_data = {
                                'rollout_id': rollout_idx,
                                'position': int(position),
                                'entropy_at_position': entropies[position],
                                'output_text': output_text,
                                'score': score,
                            }
                            rollouts.append(rollout_data)
                            
                            print(f"Rollout {rollout_idx + 1}/8 at position {position} (entropy={entropies[position]:.3f}) - Score: {score:.3f}")
                    
                    print(f"\nEvaluation Results for Problem {i+1}, Position {position} ({entropy_class} entropy):")
                    print(f"Original entropy: {entropies[position]:.3f}")
                    
                    # Store results for this position
                    position_results = {
                        'problem_id': i,
                        'position': int(position),
                        'original_entropy': entropies[position],
                        'rollouts': rollouts,
                        'original_problem': result['problem'],
                        'entropy_class': entropy_class,  # Now dynamically set based on current category
                        'expected_answer': result.get('expected_answer', None)
                    }
                    
                    # Save individual position results with entropy class in filename
                    position_output_path = dataset_path.parent / "rollouts" / f"rollout_results_problem_{i}_pos_{position}_{entropy_class}.json"
                    position_output_path.parent.mkdir(exist_ok=True)  # Create rollouts directory if it doesn't exist
                    with open(position_output_path, 'w') as f:
                        json.dump(position_results, f, indent=2, ensure_ascii=False)
                    
                    print(f"Saved rollout results to {position_output_path}")
                    
                    # Clear memory before next position
                    del rollouts, output_text
                    torch.cuda.empty_cache()





    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    def int_or_str(x):
        try:
            return int(x)
        except:
            return x

    parser.add_argument('--dataset_path', type=str, default=None)
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size to benchmark with')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--top_p', type=float, default=1.0, help='Top-p for nucleus sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--generate_initial_rollouts', action='store_true', help='Whether to generate initial rollouts.')
    parser.add_argument('--device', type=str, default=default_device, help='Device to use')

    args = parser.parse_args()
    main(
        args.dataset_path, args.num_samples, args.max_new_tokens, args.batch_size, args.top_k,
        args.top_p, args.temperature, args.checkpoint_path, args.compile, args.compile_prefill, 
        args.draft_checkpoint_path, args.speculate_k, args.generate_initial_rollouts, args.device
    )
 