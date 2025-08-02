MODEL_REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
DRAFT_MODEL_REPO=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# Use single quotes to prevent shell interpretation of special characters
PROMPT='The expression $2\cdot 3 \cdot 4\cdot 5+1$ is equal to 121, since multiplication is carried out before addition. However, we can obtain values other than 121 for this expression if we are allowed to change it by inserting parentheses. For example, we can obtain 144 by writing \[
(2\cdot (3\cdot 4)) \cdot (5+1) = 144.
\]In total, how many values can be obtained from the expression $2\cdot 3\cdot 4 \cdot 5 + 1$ by inserting parentheses? (Note that rearranging terms is not allowed, only inserting parentheses).'

python entropy_diagnostics.py \
  --checkpoint_path "checkpoints/$MODEL_REPO/model.pth" \
  --prompt "$PROMPT" \
  --max_new_tokens 200 \
  --temperature 1.0 \
  --top_k 200 \
  --save_diagnostics "entropy_analysis.json"