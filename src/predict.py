import torch
import string
import os
import unicodedata
from transformers import RobertaTokenizer, RobertaForMaskedLM
from collections import Counter, defaultdict

# Define the output path for predictions
output_path = "../output/pred.txt"

# Define valid character sets
VALID_CHARS = set(string.ascii_letters + string.digits + string.punctuation + ' ')

class RobertaPredictor:
    def __init__(self, model_dir="./work/model.checkpoint"):
        self.model_dir = model_dir
        
        # Try to load fine-tuned model if available
        self.tokenizer = None
        self.model = None
        try:
            if os.path.exists(model_dir) and os.path.isdir(model_dir):
                self.tokenizer = RobertaTokenizer.from_pretrained(model_dir)
                self.model = RobertaForMaskedLM.from_pretrained(model_dir)
                print(f"Loaded fine-tuned RoBERTa model from {model_dir}")
            else:
                # Fall back to pre-trained model
                self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
                self.model = RobertaForMaskedLM.from_pretrained("roberta-base")
                print("Using pre-trained RoBERTa model")
            
            self.model.eval()
        except Exception as e:
            print(f"Error loading RoBERTa model: {e}")
            print("Will use n-gram fallback only")
        
        # N-gram models for fallback
        self.ngram_model = {
            1: Counter(),  # Unigrams
            2: defaultdict(Counter),  # Bigrams
            3: defaultdict(Counter),  # Trigrams
        }
        
        # Load n-gram data if available
        self.load_ngram_data()
    
    def load_ngram_data(self):
        """Load n-gram data from training file if available"""
        try:
            if os.path.exists('./work/train_data.txt'):
                print("Loading training data for n-gram model")
                with open('./work/train_data.txt', 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.update_ngrams(line)
                print("N-gram model updated with training data")
        except Exception as e:
            print(f"Warning: Could not load training data for n-grams: {e}")
    
    def update_ngrams(self, text):
        """Update n-gram models with new text"""
        # Update unigrams
        for char in text:
            self.ngram_model[1][char] += 1
        
        # Update bigrams
        for i in range(len(text) - 1):
            context = text[i]
            next_char = text[i + 1]
            self.ngram_model[2][context][next_char] += 1
        
        # Update trigrams
        for i in range(len(text) - 2):
            context = text[i:i+2]
            next_char = text[i + 2]
            self.ngram_model[3][context][next_char] += 1
    
    def is_valid_char(self, char):
        """Check if a character is valid for prediction"""
        if not char or len(char) != 1:
            return False
            
        # Check if it's in our valid set
        if char in VALID_CHARS:
            return True
            
        # Additional basic checks
        if char.isalpha() or char.isdigit() or char.isspace():
            return True
            
        # Common punctuation might not be caught above
        if char in ".,;:!?'\"()[]{}-_+=<>/\\@#$%^&*":
            return True
            
        return False
    
    def predict_with_roberta(self, text, n=3):
        """Use RoBERTa model to predict next characters"""
        if self.tokenizer is None or self.model is None:
            return []
        
        try:
            # Add mask token for prediction
            masked_text = text + self.tokenizer.mask_token
            inputs = self.tokenizer(masked_text, return_tensors="pt")
            
            # Find position of mask token
            mask_token_index = (inputs.input_ids == self.tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = outputs.logits[0, mask_token_index, :]
            
            # Get probabilities
            probs = torch.softmax(predictions, dim=-1)
            
            # Get top token indices
            top_token_indices = torch.argsort(probs, descending=True)[0]
            
            # Convert to characters
            characters = []
            checked_chars = 0
            for token_id in top_token_indices:
                # Limit how many tokens we check to avoid excessive computation
                checked_chars += 1
                if checked_chars > 100:  # Check at most 100 tokens
                    break
                    
                token = self.tokenizer.decode([token_id]).strip()
                if len(token) == 1 and self.is_valid_char(token):
                    if token not in characters:
                        characters.append(token)
                if len(characters) >= n:
                    break
            
            return characters
        except Exception as e:
            print(f"Error in RoBERTa prediction: {e}")
            return []
    
    def predict_with_ngrams(self, text, n=3):
        """Use n-gram models to predict next characters"""
        predictions = []
        
        # Use trigrams
        if len(text) >= 2:
            context = text[-2:]
            for char, _ in self.ngram_model[3][context].most_common(n*2):
                if self.is_valid_char(char):
                    predictions.append(char)
                if len(predictions) >= n:
                    break
        
        # Use bigrams if needed
        if len(predictions) < n and len(text) >= 1:
            context = text[-1]
            for char, _ in self.ngram_model[2][context].most_common(n*2):
                if self.is_valid_char(char) and char not in predictions:
                    predictions.append(char)
                if len(predictions) >= n:
                    break
        
        # Use unigrams if needed
        if len(predictions) < n:
            for char, _ in self.ngram_model[1].most_common(n*2):
                if self.is_valid_char(char) and char not in predictions:
                    predictions.append(char)
                if len(predictions) >= n:
                    break
        
        return predictions[:n]
    
    def predict_next_char(self, text, n=3):
        """Predict the next n most likely characters using all available methods"""
        # Try RoBERTa first
        roberta_predictions = self.predict_with_roberta(text, n)
        
        # If RoBERTa predictions are available, use them
        if len(roberta_predictions) == n:
            return roberta_predictions
        
        # Otherwise combine with n-gram predictions
        ngram_predictions = self.predict_with_ngrams(text, n)
        
        # Combine predictions, prioritizing RoBERTa
        combined_predictions = []
        combined_predictions.extend(roberta_predictions)
        
        for char in ngram_predictions:
            if char not in combined_predictions:
                combined_predictions.append(char)
            if len(combined_predictions) >= n:
                break
        
        # If we still don't have enough, add common characters
        if len(combined_predictions) < n:
            for char in 'etaoinshrdlu':
                if char not in combined_predictions:
                    combined_predictions.append(char)
                if len(combined_predictions) >= n:
                    break
        
        return combined_predictions[:n]

def predict(pred_data):
    """Main prediction function called from the model"""
    predictor = RobertaPredictor()
    
    output = []
    for text in pred_data:
        # Get predictions
        predictions = predictor.predict_next_char(text, n=3)
        
        # Ensure we have exactly 3 predictions and they're all valid
        valid_predictions = [p for p in predictions if predictor.is_valid_char(p)]
        
        # If we don't have enough valid predictions, add common letters
        while len(valid_predictions) < 3:
            for c in 'etaoinshrdlu':
                if c not in valid_predictions:
                    valid_predictions.append(c)
                    break
        
        valid_predictions = valid_predictions[:3]  # Ensure exactly 3
        
        print(f"predicted {valid_predictions}")
        output.append(''.join(valid_predictions))
        
        # Update n-gram models for future predictions
        predictor.update_ngrams(text)
    
    # Write predictions to output
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            for pred in output:
                f.write(f"{pred}\n")
    except Exception as e:
        print(f"Warning: Could not write to {output_path}: {e}")
    
    return output

# Optional: If run directly, provide a simple interactive mode
if __name__ == "__main__":
    predictor = RobertaPredictor()
    
    print("Interactive Character Prediction")
    print("Type characters and see predictions (Ctrl+C to exit)")
    
    text = ""
    try:
        while True:
            predictions = predictor.predict_next_char(text)
            print(f"Current text: '{text}'")
            print(f"Predicted next characters: {predictions}")
            
            next_char = input("Enter next character: ")
            if next_char:
                text += next_char[0]
                # Learn from this input
                predictor.update_ngrams(next_char[0])
    except KeyboardInterrupt:
        print("\nExit")