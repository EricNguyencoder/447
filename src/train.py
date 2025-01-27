import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguements, Trainer
from dataset import load_dataset
from peft import LoraConfig, get_peft_model # used for lora fine tuning

# starts from scratch right now
model_name = "name"
output_path = f"./work/{model_name}-ft"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Dont know the format of the dataset yet

tokenized_dataset = blank

# Define LoRA configuration
lora_config = LoraConfig(
    r=8,  # increase to add more precision, -> but slow training 
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,  
    bias="none",  
    task_type="CAUSAL_LM"  
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
    evaluation_strategy="steps",  # Evaluate every `eval_steps`
    eval_steps=500,  # Evaluation frequency
    load_best_model_at_end=True,  # Load the best model at the end
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    eval_dataset=tokenized_dataset["validation"],
    tokenizer=tokenizer,
)

trainer.train()

model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

