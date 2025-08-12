MODEL_REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
DRAFT_MODEL_REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
PROMPT="How many positive whole-number divisors does 196 have?"

time python generate.py \
  --checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model.pth \
  --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model.pth \
  --speculate_k 10 \
  --prompt "$PROMPT" \
  --max_new_tokens 16000 \
  --num_samples 1 \
  --compile_prefill \
  --compile \