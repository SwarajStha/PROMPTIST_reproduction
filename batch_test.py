import torch
import gc
import os
import json
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

PROMPTS = [
    'A cat in the rain',
    'A fantasy castle',
    'Fear',
    'A cyberpunk detective standing in neon rain with a robotic arm',
    'A beautiful sunset over the mountains',
    'Portrait of a warrior',
    'The starry night by Van Gogh'
]

OUTPUT_DIR = "batch_results"

def phase_1_optimization():
    print("\n=== PHASE 1: Batch Text Optimization ===")
    model_path = "./checkpoints/promptist_sft"
    
    # Quantization Config
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
    print(f"Loading Promptist from {model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        device_map='auto',
        quantization_config=quant_config,
    )
    
    results = []
    
    print("Optimizing prompts...")
    for original in tqdm(PROMPTS):
        input_text = original.strip() + " Rephrase:"
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
            optimized = output_text.split("Rephrase:")[-1].strip()
        else:
            optimized = output_text.strip()
            
        results.append({
            "original": original,
            "optimized": optimized
        })
        
    print("\nOptimization Results:")
    for res in results:
        print(f"  Orig: {res['original']}\n  Opt:  {res['optimized']}\n")
        
    # Cleanup
    print("Unloading Promptist model...")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    
    return results

def phase_2_generation(prompt_data):
    print("\n=== PHASE 2: Batch Image Generation ===")
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"Loading {model_id}...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
    # Disable safety checker for batch processing stability (optional prompt triggers)
    pipe.safety_checker = None 
    
    print("Generating images...")
    for idx, data in enumerate(tqdm(prompt_data)):
        orig = data['original']
        opt = data['optimized']
        
        # Original
        image_orig = pipe(orig).images[0]
        orig_path = os.path.join(OUTPUT_DIR, f"prompt{idx+1}_original.png")
        image_orig.save(orig_path)
        
        # Optimized
        image_opt = pipe(opt).images[0]
        opt_path = os.path.join(OUTPUT_DIR, f"prompt{idx+1}_optimized.png")
        image_opt.save(opt_path)
        
    print(f"\nAll images saved to {OUTPUT_DIR}/")

if __name__ == "__main__":
    # 1. Optimize
    prompt_data = phase_1_optimization()
    
    # 2. Generate
    phase_2_generation(prompt_data)
