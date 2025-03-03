import torch
from torch.distributions.categorical import Categorical
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

model_checkpoint = f"./work/model.checkpoint"
output_path = "./output/pred.txt"

def predict(pred_data, batch_size = 32):
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
    model = AutoModelForCausalLM.from_pretrained(model_checkpoint)

    # Tokenize input
    output = []
    model.eval()

    # Move model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)


    for i in tqdm(range(0, len(pred_data), batch_size), desc="Processing Batches"):
        batch = pred_data[i : i + batch_size]  # Get batch

        # Tokenize all inputs at once (batch processing)
        encoded_inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True)
        input_ids = encoded_inputs["input_ids"].to(device)

        # Generate predictions in batch
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits
        

        # Extract logits for the next token
        next_tok_scores = logits[:, -1, :]

        # Convert to probabilities
        probs = torch.softmax(next_tok_scores, dim=-1)
            
        # Get top 3 tokens
        top_k_probs, top_k_tokens = torch.topk(probs, k=3, dim=-1)

        # Decode tokens
        predicted_tokens = [[tokenizer.decode(token_id) for token_id in seq] for seq in top_k_tokens]
        
        output.append(predicted_tokens)
    print(f"output is {output}")
    # write the predictions to output
    with open(output_path, 'wt') as f:
        for x in output:
            for s in x:
                f.write('{}'.format(s))
            f.write('\n')
    return output

pred_sample = ["Happ", "Happy Ne","Happy New Yea"]
predict(pred_sample)
