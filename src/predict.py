import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_checkpoint = f"./work/{model_name}-ft"
output_path = "./output/preds.txt"

tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

# Tokenize input
output = []
for i in pred_data:
    input_ids = tokenizer.encode(pred_data[i], return_tensors="pt")

    # Generate predictions
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits

    # Extract logits for the next token
    next_tok_scores = logits[0, -1, :]

    # Convert to probabilities
    probs = torch.softmax(next_tok_scores, dim=-1)

    # Get top 3 tokens
    top_k_probs, top_k_tokens = torch.topk(probs, k=3)

    # Decode tokens
    predicted_tokens = [tokenizer.decode(token_id) for token_id in top_k_tokens]

    # save to output list
    output.extend(f"{predicted_tokens[0]}{predicted_tokens[1]}{predicted_tokens[2]}")

# write the predictions to output
with open(output_path, 'wt') as f:
    for x in output:
        f.write('{}\n'.format(p))


