# Shuttle-Jaguar Image Generator

A simple command-line interface for generating images with the shuttleai/shuttle-jaguar text-to-image model using 🧨 Diffusers.

## About Shuttle-Jaguar

Shuttle-Jaguar is a powerful text-to-image generation model by shuttleai that can create high-quality images from text prompts.

## Model Weights

The model weights for shuttle-jaguar are available in FP8 format:
- Direct link: [shuttle-jaguar-fp8.safetensors](https://huggingface.co/shuttleai/shuttle-jaguar/resolve/main/fp8/shuttle-jaguar-fp8.safetensors)

When using the diffusers pipeline as implemented in this repository, the weights will be automatically downloaded from the Hugging Face Hub. However, you can also manually download the weights if needed for custom implementations or offline use.

## Installation

1. Clone this repository:
```bash
git clone https://github.com/MushroomFleet/Jaguar-Diffusers
cd Jaguar-Diffusers
```

2. Install required dependencies:
```bash
pip install -U diffusers torch transformers
```

## Usage

Generate an image using a text prompt:

```bash
python generate.py --prompt "A cat holding a sign that says hello world"
```

The generated image will be saved as `output.png` in the current directory by default.

### Command-line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--prompt` | string | (Required) | Text prompt for image generation |
| `--output` | string | `output.png` | Output image filename |
| `--height` | int | 1024 | Image height in pixels |
| `--width` | int | 1024 | Image width in pixels |
| `--guidance-scale` | float | 3.5 | Classifier-free guidance scale |
| `--steps` | int | 4 | Number of inference steps |
| `--max-seq-length` | int | 256 | Maximum sequence length for text encoding |
| `--seed` | int | (None) | Random seed for reproducibility |
| `--save-vram` | flag | (False) | Enable CPU offloading to save VRAM |
| `--compile` | flag | (False) | Enable torch.compile for performance boost |
| `--weights-path` | string | (None) | Path to local model weights file (e.g., shuttle-jaguar-fp8.safetensors) |

### Examples

#### Basic Usage
```bash
python generate.py --prompt "A beautiful mountain landscape at sunset"
```

#### Customizing Output
```bash
python generate.py --prompt "A futuristic city with flying cars" --output "future_city.png"
```

#### Changing Image Dimensions
```bash
python generate.py --prompt "A portrait of an astronaut" --width 768 --height 1024
```

#### Using a Specific Seed for Reproducibility
```bash
python generate.py --prompt "An oil painting of a dog in a park" --seed 42
```

#### Saving VRAM
```bash
python generate.py --prompt "A fantasy castle on a floating island" --save-vram
```

#### Using Local Weights File
```bash
# First download the weights
wget https://huggingface.co/shuttleai/shuttle-jaguar/resolve/main/fp8/shuttle-jaguar-fp8.safetensors

# Then use the local weights file
python generate.py --prompt "A detailed painting of a galaxy" --weights-path shuttle-jaguar-fp8.safetensors
```

## Requirements

- Python 3.7 or higher
- PyTorch 1.10 or higher
- CUDA-compatible GPU (recommended)

## License

This project is distributed under the MIT License. See the LICENSE file for more information.
