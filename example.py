#!/usr/bin/env python
"""
Example script that demonstrates how to use the shuttle-jaguar model
directly in Python code without using the command-line interface.
"""

import torch
from diffusers import DiffusionPipeline
from PIL import Image


def generate_image(
    prompt: str,
    output_path: str = "output.png",
    height: int = 1024,
    width: int = 1024,
    guidance_scale: float = 3.5,
    num_steps: int = 4,
    max_seq_length: int = 256,
    seed: int = None,
    save_vram: bool = False,
    use_compile: bool = False,
    weights_path: str = None,
):
    """
    Generate an image using the shuttle-jaguar model.

    Args:
        prompt (str): Text prompt for image generation
        output_path (str, optional): Path to save the output image. Defaults to "output.png".
        height (int, optional): Image height in pixels. Defaults to 1024.
        width (int, optional): Image width in pixels. Defaults to 1024.
        guidance_scale (float, optional): Classifier-free guidance scale. Defaults to 3.5.
        num_steps (int, optional): Number of inference steps. Defaults to 4.
        max_seq_length (int, optional): Maximum sequence length for text encoding. Defaults to 256.
        seed (int, optional): Random seed for reproducibility. Defaults to None.
        save_vram (bool, optional): Enable CPU offloading to save VRAM. Defaults to False.
        use_compile (bool, optional): Enable torch.compile for performance boost. Defaults to False.
        weights_path (str, optional): Path to local model weights file. Defaults to None.

    Returns:
        PIL.Image.Image: The generated image
    """
    import os

    # Check for CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Load the diffusion pipeline
    if weights_path and os.path.exists(weights_path):
        print(f"Loading model from local weights: {weights_path}")
        pipe = DiffusionPipeline.from_pretrained(
            "shuttleai/shuttle-jaguar", 
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            local_files_only=True,
            use_safetensors=True,
            local_files_only_safetensors_path=weights_path
        ).to(device)
    else:
        print("Loading model from HuggingFace Hub (FP8 format)")
        pipe = DiffusionPipeline.from_pretrained(
            "shuttleai/shuttle-jaguar", 
            torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            variant="fp8"
        ).to(device)
    
    # Apply VRAM saving if requested
    if save_vram:
        pipe.enable_model_cpu_offload()
    
    # Apply torch.compile optimizations if requested
    if use_compile and device == "cuda":
        pipe.transformer.to(memory_format=torch.channels_last)
        pipe.transformer = torch.compile(
            pipe.transformer, mode="max-autotune", fullgraph=True
        )
    
    # Set up seed for reproducibility if provided
    generator = None
    if seed is not None:
        generator = torch.Generator(device).manual_seed(seed)
    
    # Generate the image
    image = pipe(
        prompt,
        height=height,
        width=width,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        max_sequence_length=max_seq_length,
        generator=generator
    ).images[0]
    
    # Save the generated image
    if output_path:
        image.save(output_path)
        print(f"Image saved to: {output_path}")
    
    return image


if __name__ == "__main__":
    # Example usage
    image = generate_image(
        prompt="A serene lake surrounded by mountains at dawn, with mist rising from the water",
        output_path="example_output.png",
        # Uncomment to use custom parameters
        # height=768,
        # width=1024,
        # guidance_scale=4.0,
        # num_steps=6,
        # seed=42,
        # save_vram=True,
    )
    
    # You can also process the image further using PIL
    # For example, to show the image:
    # image.show()
    
    # Or resize it:
    # resized_image = image.resize((512, 512))
    # resized_image.save("resized_output.png")
