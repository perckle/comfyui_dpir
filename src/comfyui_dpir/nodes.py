import torch
import comfy.utils
from comfy import model_management
import torch.nn.functional as F


class ImageDenoiseWithDPIR:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "upscale_model": ("UPSCALE_MODEL",),
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 1000}),
                "tilesize": ("INT", {"default": 512, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "denoise"
    CATEGORY = "image/denoising"

    def denoise(self, upscale_model, image, strength, tilesize):
        device = model_management.get_torch_device()
        model = upscale_model.model
        model.to(device)

        if image.ndim == 3:
            image = image.unsqueeze(-1)

        input_channels = image.shape[3]
        model_input_channels = upscale_model.input_channels
        model_output_channels = upscale_model.output_channels

        image_processed = image
        if model_input_channels != input_channels:
            if model_input_channels == 1 and input_channels == 3:
                image_processed = (
                    image[:, :, :, 0] * 0.299
                    + image[:, :, :, 1] * 0.587
                    + image[:, :, :, 2] * 0.114
                )
                image_processed = image_processed.unsqueeze(-1)
            elif model_input_channels == 3 and input_channels == 1:
                image_processed = image.repeat(1, 1, 1, 3)
            else:
                raise ValueError(
                    f"Cannot convert image with {input_channels} channels to model's expected {model_input_channels} channels."
                )

        in_img = image_processed.movedim(-1, -3).to(device)

        size_multiple_of = getattr(upscale_model.size_requirements, "multiple_of", 8)
        noise_level = strength / 255.0

        def model_fn(img_tile):
            orig_H, orig_W = img_tile.shape[2], img_tile.shape[3]
            pad_H = (size_multiple_of - orig_H % size_multiple_of) % size_multiple_of
            pad_W = (size_multiple_of - orig_W % size_multiple_of) % size_multiple_of

            padded_tile = (
                F.pad(img_tile, (0, pad_W, 0, pad_H), "replicate")
                if pad_H > 0 or pad_W > 0
                else img_tile
            )

            _B, _C, pH, pW = padded_tile.shape
            noise_map = torch.full(
                (_B, 1, pH, pW),
                noise_level,
                device=padded_tile.device,
                dtype=padded_tile.dtype,
            )
            model_input = torch.cat((padded_tile, noise_map), dim=1)

            model_output = model(model_input)

            return (
                model_output[:, :, :orig_H, :orig_W]
                if pad_H > 0 or pad_W > 0
                else model_output
            )

        tile = tilesize
        overlap = 32
        oom = True

        while oom:
            try:
                steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
                    in_img.shape[3],
                    in_img.shape[2],
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                )
                pbar = comfy.utils.ProgressBar(steps)
                s = comfy.utils.tiled_scale(
                    in_img,
                    model_fn,
                    tile_x=tile,
                    tile_y=tile,
                    overlap=overlap,
                    upscale_amount=1.0,
                    pbar=pbar,
                )
                oom = False
            except model_management.OOM_EXCEPTION as e:
                tile //= 2
                if tile < 128:
                    raise e

        model.to("cpu")
        s = torch.clamp(s.movedim(-3, -1), min=0.0, max=1.0)

        if model_output_channels == 1 and s.shape[-1] == 1:
            s = s.repeat(1, 1, 1, 3)

        return (s,)


NODE_CLASS_MAPPINGS = {
    "ImageDenoiseWithDPIR": ImageDenoiseWithDPIR,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageDenoiseWithDPIR": "Denoise with DRUNet",
}
