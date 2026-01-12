import torch
import gc
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline

def optimize_prompt_phase(prompt_input):
    print("\n=== PHASE 1: Text Optimization ===")
    model_path = "./checkpoints/promptist_sft"
    print(f"Loading Promptist from {model_path}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map='auto',
        quantization_config=quant_config,
    )
    
    print("Optimizing prompt...")
    input_text = prompt_input.strip() + " Rephrase:"
    inputs = tokenizer(input_text, return_tensors='pt')
    input_ids = inputs.input_ids.to("cuda")
    attention_mask = inputs.attention_mask.to("cuda")
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=40,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True 
    )
    
    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    if "Rephrase:" in output_text:
        optimized_prompt = output_text.split("Rephrase:")[-1].strip()
    else:
        optimized_prompt = output_text.strip()
        
    print(f"Original: {prompt_input}")
    print(f"Optimized: {optimized_prompt}")
    
    return optimized_prompt, model, tokenizer

def cleanup_phase(model, tokenizer):
    print("\n=== PHASE 2: Cleanup ===")
    print("Deleting model and tokenizer...")
    del model
    del tokenizer
    
    print("Running garbage collection and emptying CUDA cache...")
    gc.collect()
    torch.cuda.empty_cache()
    print("VRAM cleared (hopefully).")

def generation_phase(original_prompt, optimized_prompt):
    print("\n=== PHASE 3: Image Generation ===")
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"Loading {model_id}...")
    
    # Load SD in float16 to save memory
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    
    print(f"Generating image for Original: '{original_prompt}'")
    image_baseline = pipe(original_prompt).images[0]
    image_baseline.save("baseline.png")
    print("Saved baseline.png")
    
    print(f"Generating image for Optimized: '{optimized_prompt}'")
    image_optimized = pipe(optimized_prompt).images[0]
    image_optimized.save("optimized.png")
    print("Saved optimized.png")

if __name__ == "__main__":
    prompt = 'A futuristic city with flying cars'
    
    # 1. Optimize
    optimized_text, model, tokenizer = optimize_prompt_phase(prompt)
    
    # 2. Cleanup
    cleanup_phase(model, tokenizer)
    
    # 3. Generate
    generation_phase(prompt, optimized_text)
    
    print("\nDone! Check baseline.png and optimized.png")
