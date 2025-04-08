import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torchtune import Tuner, TuneConfig
from torchtune.modules import LoraConfig

# Step 1: Load and prepare the Pearl Poet dataset
def prepare_pearl_poet_dataset(file_path):
    """Load Pearl Poet text and convert to training examples"""
    # Read in the manuscript text
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into stanzas based on blank lines
    stanzas = [s.strip() for s in text.split('\n\n') if s.strip()]
    
    # Create training examples
    examples = []
    for stanza in stanzas:
        lines = stanza.split('\n')
        if len(lines) >= 4:  # Ensure enough lines
            # Split at midpoint
            mid_point = len(lines) // 2
            prompt = '\n'.join(lines[:mid_point])
            completion = '\n'.join(lines[mid_point:])
            
            # Format as a training example
            examples.append({
                "text": f"{prompt}\n{completion}"
            })
    
    return Dataset.from_list(examples)

# Step 2: Setup tokenizer and model
def setup_model_and_tokenizer():
    # Set model ID (use 8B instead of 70B for reasonable compute requirements)
    model_id = "meta-llama/Meta-Llama-3-8B"
    
    # Configure quantization for efficient training
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=quant_config,
        device_map="auto"
    )
    
    return model, tokenizer

# Step 3: Configure TorchTune and LoRA
def setup_tuner(model):
    # Configure LoRA parameters
    lora_config = LoraConfig(
        r=16,                # Rank
        lora_alpha=32,       # Alpha scaling factor
        lora_dropout=0.05,   # Dropout probability
        target_modules=[     # Target specific modules for efficiency
            "q_proj", 
            "k_proj", 
            "v_proj", 
            "o_proj"
        ]
    )
    
    # Configure tuning parameters
    tune_config = TuneConfig(
        lr=2e-4,                     # Learning rate
        batch_size=4,                # Per device batch size
        num_epochs=3,                # Number of training epochs
        max_seq_length=512,          # Maximum sequence length
        gradient_accumulation=4,     # Gradient accumulation steps
        weight_decay=0.01,           # Weight decay for regularization
        optimizer="8bit_adamw",      # 8-bit AdamW optimizer
        warmup_ratio=0.03,           # Learning rate warmup
        logging_steps=10,            # Log every N steps
        save_steps=100,              # Save checkpoint every N steps
        save_total_limit=3           # Keep only the last N checkpoints
    )
    
    # Create tuner
    tuner = Tuner(
        model=model,
        tune_config=tune_config,
        lora_config=lora_config
    )
    
    return tuner

# Step 4: Fine-tune the model
def train_model(tuner, dataset, tokenizer):
    # Define preprocessing function
    def preprocess_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True,
            max_length=512
        )
    
    # Preprocess the dataset
    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=["text"]
    )
    
    # Start training
    tuner.fit(tokenized_dataset)
    
    # Save the tuned model
    output_dir = "./pearl_poet_llama3"
    tuner.save(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    return output_dir

# Step 5: Function to generate text with the fine-tuned model
def test_model(model_path, tokenizer_path, prompt):
    # Load fine-tuned model
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Load the base model
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True),
        device_map="auto"
    )
    
    # Apply the fine-tuned weights
    tuner = Tuner.from_pretrained(
        model=base_model,
        pretrained_model_path=model_path
    )
    model = tuner.model
    
    # Generate text
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True
        )
    
    # Return generated text
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Main function
def main():
    print("Step 1: Preparing Pearl Poet dataset")
    dataset = prepare_pearl_poet_dataset("pearl_poet_manuscript.txt")
    print(f"Created dataset with {len(dataset)} examples")
    
    print("Step 2: Setting up model and tokenizer")
    model, tokenizer = setup_model_and_tokenizer()
    
    print("Step 3: Setting up TorchTune")
    tuner = setup_tuner(model)
    
    print("Step 4: Fine-tuning the model")
    output_dir = train_model(tuner, dataset, tokenizer)
    print(f"Model saved to {output_dir}")
    
    print("Step 5: Testing the model")
    test_prompt = "In the fayre felde of paradys,"
    generated_text = test_model(output_dir, output_dir, test_prompt)
    print("\nGenerated Text:")
    print("-" * 50)
    print(generated_text)
    print("-" * 50)

if __name__ == "__main__":
    main()