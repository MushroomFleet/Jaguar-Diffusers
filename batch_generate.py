#!/usr/bin/env python
"""
Batch image generation script for the shuttle-jaguar model using diffusers pipeline.
This script reads prompts from a text file and generates an image for each prompt.
"""

import argparse
import os
import sys
import torch
from diffusers import DiffusionPipeline
from datetime import datetime
import time


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate multiple images with the shuttle-jaguar model using prompts from a file."
    )
    parser.add_argument(
        "--prompts-file", 
        type=str, 
        required=True, 
        help="Text file containing prompts (one per line)"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="outputs",
        help="Output directory for generated images (default: outputs)"
    )
    parser.add_argument(
        "--height", 
        type=int, 
        default=1024,
        help="Image height in pixels (default: 1024)"
    )
    parser.add_argument(
        "--width", 
        type=int, 
        default=1024,
        help="Image width in pixels (default: 1024)"
    )
    parser.add_argument(
        "--guidance-scale", 
        type=float, 
        default=3.5,
        help="Classifier-free guidance scale (default: 3.5)"
    )
    parser.add_argument(
        "--steps", 
        type=int, 
        default=4,
        help="Number of inference steps (default: 4)"
    )
    parser.add_argument(
        "--max-seq-length", 
        type=int, 
        default=256,
        help="Maximum sequence length for text encoding (default: 256)"
    )
    parser.add_argument(
        "--base-seed", 
        type=int,
        help="Base random seed for reproducibility (will be incremented for each prompt)"
    )
    parser.add_argument(
        "--save-vram", 
        action="store_true",
        help="Enable CPU offloading to save VRAM"
    )
    parser.add_argument(
        "--compile", 
        action="store_true",
        help="Enable torch.compile for potential performance boost (may increase loading time)"
    )
    parser.add_argument(
        "--filename-prefix", 
        type=str, 
        default="image",
        help="Prefix for the output filenames (default: 'image')"
    )
    
    return parser.parse_args()


def read_prompts_file(file_path):
    """Read prompts from a text file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            # Strip whitespace and filter out empty lines
            prompts = [line.strip() for line in f.readlines()]
            prompts = [p for p in prompts if p and not p.startswith('#')]
            
        if not prompts:
            raise ValueError("The prompts file is empty or contains only comments.")
        
        return prompts
    except Exception as e:
        print(f"Error reading prompts file: {str(e)}")
        sys.exit(1)


def generate_filename(prefix, index, prompt):
    """Generate a filename for the output image."""
    # Create a short slug from the prompt (first 30 chars, no special chars)
    slug = ''.join(c if c.isalnum() else '_' for c in prompt[:30]).rstrip('_')
    
    # Return the filename with index and slug
    return f"{prefix}_{index:03d}_{slug}.png"


def main():
    """Main function to run the batch image generation pipeline."""
    args = parse_arguments()
    
    # Read prompts from file
    prompts = read_prompts_file(args.prompts_file)
    print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available. Using CPU for inference (this will be slow).")
    
    print(f"Loading shuttle-jaguar model...")
    
    try:
        # Load the diffusion pipeline with the appropriate settings
        pipe = DiffusionPipeline.from_pretrained(
            "shuttleai/shuttle-jaguar", 
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32
        ).to(device)
        
        # Apply VRAM saving if requested
        if args.save_vram:
            print("Enabling CPU offloading to save VRAM...")
            pipe.enable_model_cpu_offload()
        
        # Apply torch.compile optimizations if requested
        if args.compile and device == "cuda":
            print("Applying torch.compile optimizations (this may take a while)...")
            pipe.transformer.to(memory_format=torch.channels_last)
            pipe.transformer = torch.compile(
                pipe.transformer, mode="max-autotune", fullgraph=True
            )
        
        # Process each prompt
        total_start_time = datetime.now()
        for i, prompt in enumerate(prompts):
            print(f"\nProcessing prompt {i+1}/{len(prompts)}: {prompt}")
            
            # Set up seed for reproducibility if provided
            generator = None
            if args.base_seed is not None:
                seed = args.base_seed + i
                print(f"Using seed: {seed}")
                generator = torch.Generator(device).manual_seed(seed)
            
            # Generate output filename
            output_filename = generate_filename(args.filename_prefix, i+1, prompt)
            output_path = os.path.join(args.output_dir, output_filename)
            
            # Print generation parameters
            print(f"Generating with parameters:")
            print(f"  Dimensions: {args.width}x{args.height}")
            print(f"  Guidance scale: {args.guidance_scale}")
            print(f"  Inference steps: {args.steps}")
            
            # Generate the image
            start_time = datetime.now()
            image = pipe(
                prompt,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.steps,
                max_sequence_length=args.max_seq_length,
                generator=generator
            ).images[0]
            end_time = datetime.now()
            
            # Save the generated image
            image.save(output_path)
            print(f"Image generated in {(end_time - start_time).total_seconds():.2f} seconds.")
            print(f"Saved to: {output_path}")
            
            # Small delay to prevent GPU from overheating in long batches
            if i < len(prompts) - 1 and device == "cuda":
                time.sleep(0.5)
                
        total_end_time = datetime.now()
        print(f"\nBatch processing complete!")
        print(f"Total time: {(total_end_time - total_start_time).total_seconds():.2f} seconds")
        print(f"Generated {len(prompts)} images in {args.output_dir}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
