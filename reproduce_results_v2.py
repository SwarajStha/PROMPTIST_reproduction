import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

def load_prompter(model_path="./checkpoints/promptist_sft"):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Modern Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    # Low VRAM loading configurations
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map='auto',
        quantization_config=quant_config,
    )
    
    # Crucial Tokenizer Fix
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return model, tokenizer

def optimize_prompt(model, tokenizer, user_input):
    input_text = user_input.strip() + " Rephrase:"
    
    # Tokenize and move to CUDA
    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    
    # Generate optimization
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=40,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True 
    )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract just the rephrased part
    if "Rephrase:" in output_text:
        res = output_text.split("Rephrase:")[-1].strip()
    else:
        res = output_text.strip()
        
    return res

if __name__ == "__main__":
    model, tokenizer = load_prompter()
    
    test_input = "A futuristic city with flying cars"
    print(f"\nOriginal Prompt: {test_input}")
    
    optimized = optimize_prompt(model, tokenizer, test_input)
    print(f"Optimized Prompt: {optimized}")
