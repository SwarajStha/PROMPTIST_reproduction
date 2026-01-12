import torch
import os
import json
from diffusers import StableDiffusionPipeline
from tqdm import tqdm

INPUT_FILE = "batch_results/prompts.json"
OUTPUT_DIR = "batch_results_v2"

def generate_images_from_json():
    print("\n=== Batch Image Generation v2 ===")
    
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found. Please run save_batch_prompts.py first.")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # Load prompts
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        prompt_data = json.load(f)
        
    print(f"Loaded {len(prompt_data)} prompt pairs from {INPUT_FILE}")

    # Load Model
    model_id = "runwayml/stable-diffusion-v1-5"
    print(f"Loading {model_id}...")
    
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")
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
    generate_images_from_json()
