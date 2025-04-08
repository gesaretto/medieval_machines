import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

# Load original Pearl Poet text data
def load_pearl_poet_data(file_path):
    """Load and prepare Pearl Poet text data from file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into stanzas (adjust delimiter based on your formatting)
    stanzas = [s.strip() for s in text.split('\n\n') if s.strip()]
    
    # Create completion examples (first half -> second half)
    examples = []
    for stanza in stanzas:
        lines = stanza.split('\n')
        if len(lines) >= 4:  # Ensure enough lines for splitting
            # Create a training example with first lines as prompt, rest as completion
            mid_point = len(lines) // 2
            prompt = '\n'.join(lines[:mid_point])
            completion = '\n'.join(lines[mid_point:])
            examples.append({"prompt": prompt, "completion": completion})
    
    return examples

# Set up the model and tokenizer
def setup_model():
    """Set up the model with quantization for efficient fine-tuning"""
    model_id = "mistralai/Mistral-7B-v0.1"  # or "meta-llama/Llama-2-7b-hf"
    
    # Quantization config
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )
    
    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    
    # Add special tokens if needed
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Prepare model for LoRA training
    model = prepare_model_for_kbit_training(model)
    
    # LoRA config
    lora_config = LoraConfig(
        r=16,              # Rank
        lora_alpha=32,     # Alpha parameter
        lora_dropout=0.05, # Dropout probability
        bias="none",       # Bias type
        task_type="CAUSAL_LM", # Task type
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]  # Target modules to fine-tune
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, lora_config)
    
    return model, tokenizer

# Set up training
def train_model(data, model, tokenizer):
    """Train the model on Pearl Poet data"""
    # Convert data to dataset
    dataset = Dataset.from_list(data)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./pearl_poet_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        save_steps=100,
        logging_steps=20,
        learning_rate=2e-4,
        fp16=True,
        optim="paged_adamw_8bit",
        max_grad_norm=0.3,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        report_to="none"
    )
    
    # Set up trainer
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        dataset_text_field="prompt",
        max_seq_length=512,
    )
    
    # Train model
    trainer.train()
    
    # Save model
    trainer.save_model("./pearl_poet_model_final")
    
    return trainer

# Main function
def main():
    # Path to Pearl Poet text file
    data_path = "pearl_poet_manuscript.txt"
    
    # Load and prepare data
    data = load_pearl_poet_data(data_path)
    print(f"Created {len(data)} training examples")
    
    # Set up model
    model, tokenizer = setup_model()
    
    # Train model
    trainer = train_model(data, model, tokenizer)
    
    print("Fine-tuning complete!")

if __name__ == "__main__":
    main()