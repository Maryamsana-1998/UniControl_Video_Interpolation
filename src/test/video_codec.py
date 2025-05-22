import sys

if "./" not in sys.path:
    sys.path.append("./")
from utils.share import *
import utils.config as config

import einops
import numpy as np
import cv2
import torch
from pytorch_lightning import seed_everything
from annotator.util import HWC3
from models.util import create_model, load_state_dict
from models.ddim_hacked import DDIMSampler
import numpy as np
import os
import re

# ====== UTILITIES ======
def get_png_paths(directory):
    """Return a sorted list of all .png file paths in a directory."""
    if not directory or not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith('.png')
    )

def frame_number(path):
    name = os.path.basename(path)
    match = re.search(r'\d{4}', name)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No 4-digit frame number found in filename: {name}")

def select_intra_frames_by_gop(paths, gop_size):
    """Select only those intra-frame paths whose frame number % gop_size == 0."""

    sorted_paths = sorted(paths, key=frame_number)
    return [
        p for p in sorted_paths
        if frame_number(p) % gop_size == 0
    ]

def process(
    model,
    local_images,
    prompt,
    a_prompt="best quality, extremely detailed",
    n_prompt="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality",
    num_samples=1,
    image_resolution=512,
    ddim_steps=50,
    strength=1,
    scale=7.5,
    seed=42,
    eta=0.0,
    global_strength=1
):


    seed_everything(seed)
    ddim_sampler = DDIMSampler(model)

    with torch.no_grad():
        W, H = image_resolution, image_resolution

        local_conditions=[]

        for image in local_images:
            image = cv2.resize(image, (W, H))
            detected_map = HWC3(image)
            local_conditions.append(detected_map)
            
        content_emb = np.zeros((768))

        detected_maps = np.concatenate([condition for condition in local_conditions], axis=2)

        local_control = torch.from_numpy(detected_maps.copy()).float().cuda() / 255.0
        local_control = torch.stack([local_control for _ in range(num_samples)], dim=0)
        local_control = einops.rearrange(local_control, "b h w c -> b c h w").clone()
        global_control = torch.from_numpy(content_emb.copy()).float().cuda().clone()
        global_control = torch.stack([global_control for _ in range(num_samples)], dim=0)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        uc_local_control = local_control
        uc_global_control = torch.zeros_like(global_control)
        cond = {
            "local_control": [local_control],
            "c_crossattn": [
                model.get_learned_conditioning([prompt + ", " + a_prompt] * num_samples)
            ],
            "global_control": [global_control],
        }
        un_cond = {
            "local_control": [uc_local_control],
            "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)],
            "global_control": [uc_global_control],
        }
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model.low_vram_shift(is_diffusing=True)

        model.control_scales = [strength] * 13
        samples, _ = ddim_sampler.sample(
            ddim_steps,
            num_samples,
            shape,
            cond,
            verbose=True,
            eta=eta,
            unconditional_guidance_scale=scale,
            unconditional_conditioning=un_cond,
            global_strength=global_strength,
        )

        if config.save_memory:
            model.low_vram_shift(is_diffusing=False)

        x_samples = model.decode_first_stage(samples)
        x_samples = (
            (einops.rearrange(x_samples, "b c h w -> b h w c") * 127.5 + 127.5)
            .cpu()
            .numpy()
            .clip(0, 255)
            .astype(np.uint8)
        )
        results = [x_samples[i] for i in range(num_samples)]

    return (results, detected_maps)
