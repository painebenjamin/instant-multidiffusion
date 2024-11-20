import torch

from PIL import Image
from diffusers import AutoPipelineForText2Image
from multidiffusion import enable_2d_multidiffusion

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    variant="fp16"
)
# Recommended to enable model CPU offload for large images
pipeline.enable_model_cpu_offload()

# Recommended to enable VAE tiling as well for very large images
pipeline.enable_vae_tiling()

generator = torch.Generator()
seed = 123456
kwargs = {
    "prompt": "a photo of a snowy mountain peak with skiiers",
    "width": 4096,
    "height": 1024,
    "generator": generator,
    "num_inference_steps": 28,
}

# Running this size without multidiffusion peaks at ~17GB of GPU memory for SDXL
generator.manual_seed(seed)
result_no_multidiffusion = pipeline(**kwargs).images[0]
# Default tile size is 64 (512px) for SD1.5 and 128 (1024px) for everything else,
# but it tries to read from the model config if not specified
enable_2d_multidiffusion(
    pipeline.unet, # or pipeline.transformer for DiTs
    # tile_size=<int> OR tile_size=<int, int> for non-square tiles
    # tile_stride=<int> OR tile_stride=<int, int> for non-square strides
    # mask_type=(constant|bilinear|gaussian), default is bilinear
)

# Running with multidiffusion peaks at ~13GB of GPU memory for SDXL
generator.manual_seed(seed)
result_multidiffusion = pipeline(**kwargs).images[0]

collage = Image.new("RGB", (kwargs["width"], kwargs["height"] * 2))
collage.paste(result_no_multidiffusion, (0, 0))
collage.paste(result_multidiffusion, (0, kwargs["height"]))
collage.save("example-output-sdxl.jpg")
