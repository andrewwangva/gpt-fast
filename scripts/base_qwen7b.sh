export MODEL_REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
time python math_eval.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --prompt "What is 1000 * 100?" --max_new_tokens 1000 --num_samples 1  --temperature 0 --compile_prefill
