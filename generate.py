#!/usr/bin/env python
"""
Image generation script for the shuttle-jaguar model using diffusers pipeline.
"""

import argparse
import os
import sys
import torch
from diffusers import DiffusionPipeline
from datetime import datetime


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Generate images with the shuttle-jaguar model."
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        required=True, 
        help="Text prompt for image generation"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="output.png",
        help="Output image filename (default: output.png)"
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
        "--seed", 
        type=int, 
        help="Random seed for reproducibility"
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
        "--weights-path",
        type=str,
        help="Path to local model weights file (e.g., shuttle-jaguar-fp8.safetensors) if you want to use local weights instead of downloading from HuggingFace"
    )
    
    return parser.parse_args()


def main():
    """Main function to run the image generation pipeline."""
    args = parse_arguments()
    
    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Warning: CUDA not available. Using CPU for inference (this will be slow).")
    
    print(f"Loading shuttle-jaguar model...")
    
    try:
        # Load the diffusion pipeline with the appropriate settings
        if args.weights_path and os.path.exists(args.weights_path):
            print(f"Loading model from local weights: {args.weights_path}")
            pipe = DiffusionPipeline.from_pretrained(
                "shuttleai/shuttle-jaguar", 
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                local_files_only=True,
                use_safetensors=True,
                local_files_only_safetensors_path=args.weights_path
            ).to(device)
        else:
            print("Loading model from HuggingFace Hub (FP8 format)")
            pipe = DiffusionPipeline.from_pretrained(
                "shuttleai/shuttle-jaguar", 
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                variant="fp8"
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
        
        # Set up seed for reproducibility if provided
        generator = None
        if args.seed is not None:
            print(f"Using seed: {args.seed}")
            generator = torch.Generator(device).manual_seed(args.seed)
        
        # Print generation parameters
        print(f"\nGenerating image with parameters:")
        print(f"  Prompt: {args.prompt}")
        print(f"  Dimensions: {args.width}x{args.height}")
        print(f"  Guidance scale: {args.guidance_scale}")
        print(f"  Inference steps: {args.steps}")
        print(f"  Max sequence length: {args.max_seq_length}")
        
        # Generate the image
        start_time = datetime.now()
        image = pipe(
            args.prompt,
            height=args.height,
            width=args.width,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            max_sequence_length=args.max_seq_length,
            generator=generator
        ).images[0]
        end_time = datetime.now()
        
        # Save the generated image
        image.save(args.output)
        print(f"\nImage generated successfully in {(end_time - start_time).total_seconds():.2f} seconds.")
        print(f"Saved to: {os.path.abspath(args.output)}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
