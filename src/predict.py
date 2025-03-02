# import torch
# from torch.distributions.categorical import Categorical
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from transformers import BertTokenizer, BertForMaskedLM
# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# model_checkpoint = "distilgpt2"
# # model_checkpoint = f"./work/model.checkpoint"
# # model_checkpoint = "google-bert/bert-base-multilingual-cased"
# output_path = "../output/pred.txt"

# def predict(pred_data):
#     tokenizer = GPT2Tokenizer.from_pretrained(model_checkpoint)
#     model = GPT2LMHeadModel.from_pretrained(model_checkpoint)
#     model.eval()
    
#     output = []
#     for text in pred_data:
#         # Tokenize input
#         inputs = tokenizer(text, return_tensors="pt")
        
#         # Get predictions
#         with torch.no_grad():
#             outputs = model(**inputs)
#             logits = outputs.logits[0, -1]  # Get logits for next position
            
#         # Convert logits to probabilities
#         probs = torch.softmax(logits, dim=-1)
        
#         # Get single character predictions
#         char_predictions = []
#         for token_id in torch.argsort(probs, descending=True):
#             char = tokenizer.decode([token_id]).strip()
#             if len(char) == 1 and char.isprintable():
#                 char_predictions.append(char)
#             if len(char_predictions) == 3:
#                 break
                
#         # Ensure we always have 3 predictions
#         while len(char_predictions) < 3:
#             for c in 'eaio ':  # Common English characters
#                 if c not in char_predictions:
#                     char_predictions.append(c)
#                     break
        
#         print(f"predicted {char_predictions}")
#         output.append(char_predictions)

#     # Write predictions to output
#     with open(output_path, 'wt') as f:
#         for x in output:
#             for s in x:
#                 f.write('{}'.format(s))
#             f.write('\n')
#     return output
import torch
import string
from transformers import RobertaTokenizer, RobertaForMaskedLM

model_checkpoint = "roberta-base"
output_path = "../output/pred.txt"

def is_valid_char(char):
    return char in string.ascii_letters + ' '

def predict(pred_data):
    tokenizer = RobertaTokenizer.from_pretrained(model_checkpoint)
    model = RobertaForMaskedLM.from_pretrained(model_checkpoint)
    model.eval()
    
    output = []
    for text in pred_data:
        # Tokenize input
        inputs = tokenizer(text + "<mask>", return_tensors="pt")
        
        # Get predictions
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -2]  # Get logits for masked position
            
        # Convert logits to probabilities
        probs = torch.softmax(logits, dim=-1)
        
        # Get single character predictions
        char_predictions = []
        for token_id in torch.argsort(probs, descending=True):
            char = tokenizer.decode(token_id).strip()
            # Only add valid single characters
            if len(char) == 1 and is_valid_char(char):
                char_predictions.append(char)
            if len(char_predictions) == 3:
                break
                
        # If we don't have enough predictions, add common letters
        while len(char_predictions) < 3:
            for c in 'eaionrtlsu':
                if c not in char_predictions:
                    char_predictions.append(c)
                    break
        
        print(f"predicted {char_predictions}")
        output.append(char_predictions)

    # Write predictions to output
    with open(output_path, 'w', encoding='utf-8') as f:
        for x in output:
            for s in x:
                f.write(s)
            f.write('\n')
    return output