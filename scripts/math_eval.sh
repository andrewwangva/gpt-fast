export MODEL_REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
time python math_eval.py --compile --checkpoint_path checkpoints/$MODEL_REPO/model.pth --dataset_path "data/math_500_train.jsonl" --max_new_tokens 16000 --num_samples 1  --temperature 0.6 --top_p 0.9 --compile_prefill
