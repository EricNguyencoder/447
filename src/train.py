import torch
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling, AutoTokenizer
from datasets import load_dataset, Dataset
from peft import LoraConfig, get_peft_model # used for lora fine tuning

# starts from scratch right now
def train(tokenizer, model, dataset):
    print("in train")
    model_name = "name"
    output_path = f"./work/model.checkpoint"

    # 
    print("tokenizing dataset")
    tokenized_train = tokenizer(dataset, padding=True, truncation=True, return_tensors="pt")

    # Convert tokenized data into Dataset format
    train_dataset = Dataset.from_dict({
        "input_ids": tokenized_train["input_ids"],
        "attention_mask": tokenized_train["attention_mask"],
    })


    # Define LoRA configuration
    lora_config = LoraConfig(
        r=8,  # increase to add more precision, -> but slow training 
        lora_alpha=32,
        target_modules=["query", "value"],
        lora_dropout=0.1,  
        bias="none",  
        task_type="CAUSAL_LM"  
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # No Masked LM, since this is causal LM
    )

    # Apply LoRA to the model
    model = get_peft_model(model, lora_config)

    training_args = TrainingArguments(
        output_dir=output_path,  # Output directory
        per_device_train_batch_size=4,  # Batch size per device
        gradient_accumulation_steps=4,  # Gradient accumulation steps
        num_train_epochs=3,  # Number of training epochs
        learning_rate=2e-4,  # Learning rate
        fp16=True,  # Use mixed precision (if supported)
        save_steps=500,  # Save checkpoint every 500 steps
        save_total_limit=2,  # Keep only the last 2 checkpoints
        logging_dir="./logs",  # Directory for logs
        logging_steps=100,  # Log every 100 steps
        evaluation_strategy="no",  # Evaluate every `eval_steps`
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    print("starting train")

    trainer.train()

    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

tokenizer = AutoTokenizer.from_pretrained('xlm-roberta-base')
model = AutoModelForCausalLM.from_pretrained("xlm-roberta-base")

train(tokenizer, model, dataset)