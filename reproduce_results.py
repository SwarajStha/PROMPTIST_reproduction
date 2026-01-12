import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_prompter(model_path="./checkpoints/promptist_sft"):
    print(f"Loading model from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    
    # Low VRAM loading configurations
    # device_map='auto' will distribute layers to GPU/CPU if needed, but primarily GPU
    # load_in_8bit=True uses bitsandbytes for 8-bit quantization
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map='auto',
        load_in_8bit=True,
        torch_dtype=torch.float16,
    )
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    return model, tokenizer

def optimize_prompt(model, tokenizer, user_input):
    input_text = user_input.strip() + " Rephrase:"
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
    
    # Generate optimization
    # Using similar parameters to the original repo generally improves consistency
    outputs = model.generate(
        input_ids, 
        do_sample=False, 
        max_new_tokens=30, 
        num_beams=8, 
        num_return_sequences=1, 
        eos_token_id=tokenizer.eos_token_id, 
        pad_token_id=tokenizer.eos_token_id, 
        length_penalty=-1.0
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
