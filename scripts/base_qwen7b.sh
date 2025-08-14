MODEL_REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
DRAFT_MODEL_REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
PROMPT='Let $p(x)$ be a polynomial of degree 5 such that \[p(n) = \frac{n}{n^2 - 1}\]for $n = 2,$ 3, 4, $\dots,$ 7.  Find $p(8).$'
time python generate.py \
  --checkpoint_path checkpoints/$MODEL_REPO/model.pth \
  --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model.pth \
  --speculate_k 4 \
  --prompt "$PROMPT" \
  --max_new_tokens 16000 \
  --num_samples 1 \
  --temperature 0.6 \
  --top_p 0.95 \