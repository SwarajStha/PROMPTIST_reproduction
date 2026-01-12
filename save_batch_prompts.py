import torch
import json
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
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
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "prompts.json")

def generate_prompts():
    print("Loading model for text generation...")
    model_path = "./checkpoints/promptist_sft"
    
    quant_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True
    )
    
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
    
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
        
    print(f"\nSaved prompt mapping to {OUTPUT_FILE}")
    for res in results:
        print(f"Original: {res['original']}")
        print(f"Optimized: {res['optimized']}")
        print("-" * 40)

if __name__ == "__main__":
    generate_prompts()
