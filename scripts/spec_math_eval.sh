MODEL_REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
DRAFT_MODEL_REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

time python math_eval.py \
  --checkpoint_path checkpoints/$MODEL_REPO/model.pth \
  --draft_checkpoint_path checkpoints/$DRAFT_MODEL_REPO/model.pth \
  --speculate_k 8 \
  --dataset_path "data/math_500_train.jsonl" \
  --max_new_tokens 16000 \
  --num_samples 1 \
  --temperature 0.6 \
  --top_p 0.95 \
  --compile \
  --compile_prefill