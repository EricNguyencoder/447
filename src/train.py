import torch
from transformers import RobertaTokenizer, RobertaForMaskedLM
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
from datasets import Dataset
import os

def train(tokenizer=None, model=None, dataset=None):
    """
    Fine-tune RoBERTa for masked language modeling to improve character prediction.
    
    Args:
        tokenizer: The tokenizer to use (will load RobertaTokenizer if None)
        model: The model to fine-tune (will load RobertaForMaskedLM if None)
        dataset: List of text samples to train on
    
    Returns:
        None (saves model to disk)
    """
    print("Starting RoBERTa fine-tuning for character prediction")
    output_path = "./work/model.checkpoint"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize tokenizer and model if not provided
    if tokenizer is None:
        print("Loading RoBERTa tokenizer")
        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    if model is None:
        print("Loading RoBERTa model")
        model = RobertaForMaskedLM.from_pretrained("roberta-base")
    
    # Load dataset if not provided
    if dataset is None:
        try:
            print("Loading training data from ./work/train_data.txt")
            dataset = []
            with open("./work/train_data.txt", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        dataset.append(line)
        except Exception as e:
            print(f"Error loading training data: {e}")
            return
    
    if not dataset:
        print("Error: No data available for training")
        return
    
    print(f"Training on {len(dataset)} sentences")
    
    # Tokenize dataset for MLM training
    print("Tokenizing dataset")
    tokenized_texts = []
    
    for text in dataset:
        # Process each text for masked language modeling
        if text:
            try:
                # Tokenize with truncation to prevent excessively long sequences
                encoded = tokenizer(text, 
                                  truncation=True, 
                                  max_length=128,
                                  padding="max_length",
                                  return_tensors="pt")
                
                tokenized_texts.append({
                    "input_ids": encoded["input_ids"][0],
                    "attention_mask": encoded["attention_mask"][0]
                })
            except Exception as e:
                print(f"Error tokenizing text: {e}")
    
    if not tokenized_texts:
        print("Error: No texts could be tokenized")
        return
    
    print(f"Created dataset with {len(tokenized_texts)} samples")
    
    # Create dataset object
    train_dataset = Dataset.from_list(tokenized_texts)
    
    # Configure data collator for masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,               # Use masked language modeling
        mlm_probability=0.15     # Standard masking probability
    )
    
    # Set up training arguments
    training_args = TrainingArguments(
        output_dir=output_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=5e-5,
        num_train_epochs=1,      # Adjust as needed
        save_steps=500,
        save_total_limit=2,
        logging_steps=100,
        fp16=torch.cuda.is_available(),  # Use mixed precision if GPU available
        report_to="none"         # Disable reporting to external services
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Start training
    print("Starting training")
    trainer.train()
    
    # Save the model
    print(f"Saving model to {output_path}")
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print("Training completed")

# Run this if the script is executed directly
if __name__ == "__main__":
    train()