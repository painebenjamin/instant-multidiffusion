# Instant Multi-Diffusion

Inject multi-diffusion into your UNets/Transformers with two lines of code.

```py
from multidiffusion import enable_2d_multidiffusion
enable_2d_multidiffusion(pipeline.unet) # or `pipeline.transformer`
```

# What is Multi-Diffusion?

See [multidiffusion.github.io](https://multidiffusion.github.io/) for the original paper, for our purposes, it is:

1. a way to reduce memory consumption ***and*** reduce runtime <sup>(sometimes)</sup> ***and*** improve image composition <sup>(usually)</sup> when working with diffusion models at very large resolutions, and
2. a way to generate images of dynamic resolution using diffusion models that are only capable of static resolutions natively.

# Compatibility

The `enable_2d_multidiffusion` method will work on any UNet or 2D Transformer that accepts 4-dimensional Tensor input `(B×C×H×W)` and returns the same, either in latent space or pixel space.

**At present it will not work with FLUX, as that uses packed Tensor input/output.**

# Example Code

See [example-sdxl.py](https://github.com/painebenjamin/instant-multidiffusion/blob/main/example-sdxl.py) for a complete example generating the following collage using SDXL 1.0.

<div align="center">
  <a href="https://github.com/painebenjamin/instant-multidiffusion/blob/main/example-output-sdxl.jpg?raw=true" target="_blank">
    <img src="https://github.com/painebenjamin/instant-multidiffusion/blob/main/example-output-sdxl.jpg?raw=true" /><br />
    <em>SDXL 1.0: No multi-diffusion on top, multi-diffusion on bottom. Click to see full size (4096x2048)</em>
  </a>
</div>

# To-Do List

- [ ] Wrap-around (tiling images)
- [ ] Tile batching (sacrifice some memory savings for faster generation)
- [ ] FLUX compatibility
- [ ] Include example for using Multi-Diffusion for regional prompting
