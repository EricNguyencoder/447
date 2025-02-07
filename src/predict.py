import torch
from torch.distributions.categorical import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM

model_checkpoint = f"./work/model.checkpoint"
output_path = "./output/preds.txt"

def predict(pred_data):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    # Tokenize input
    output = []
    for i in range(len(pred_data)):
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
        print(f"predicted {predicted_tokens}")
        output.append(predicted_tokens)

    # write the predictions to output
    with open(output_path, 'wt') as f:
        for x in output:
            for s in x:
                f.write('{}'.format(s))
            f.write('\n')
    return output
