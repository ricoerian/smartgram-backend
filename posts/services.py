import os
import cv2
import numpy as np
import torch
from functools import lru_cache
from typing import Optional, Tuple, List
from PIL import Image

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL
)
from transformers import (
    DPTImageProcessor, 
    DPTForDepthEstimation,
    CLIPVisionModelWithProjection,
    CLIPImageProcessor
)

AI_PIPELINE = None
DEPTH_ESTIMATOR = None

BASE_MODEL_ID = "SG161222/RealVisXL_V4.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
CONTROLNET_IDS = [
    "diffusers/controlnet-canny-sdxl-1.0",
    "diffusers/controlnet-depth-sdxl-1.0",
]
IP_ADAPTER_REPO = "h94/IP-Adapter"
IP_ADAPTER_FILE = "ip-adapter-plus_sdxl_vit-h.safetensors"

MAX_IMAGE_SIZE = 1024
VIDEO_TARGET_SIZE = 576
DEFAULT_IMAGE_STEPS = 30
DEFAULT_VIDEO_STEPS = 20
FRAME_SKIP_INTERVAL = 2

STYLE_PROMPTS = {
    "auto": "raw photo, 8k uhd, dslr, soft lighting, high quality, film grain, fujifilm",
    "noir": "film noir, black and white, dramatic shadows, cinematic lighting, vintage thriller",
    "sepia": "vintage photograph, sepia tone, 1950s style, grainy texture, nostalgic",
    "sketch": "charcoal sketch, graphite drawing, rough lines, artistic shading, white paper",
    "cyber": "cyberpunk city, neon lights, futuristic fashion, rainy street, blue and pink glow",
    "hdr": "hdr photography, dramatic sky, sharp focus, vivid colors, hyperdetailed",
    "cartoon": "disney pixar style, 3d render, cute character, vibrant colors, smooth",
    "anime": "anime art, makoto shinkai style, high quality, detailed background",
    "ghibli": "studio ghibli style, watercolor background, peaceful atmosphere, whimsical",
    "realistic": "raw photo, dslr, 85mm lens, pores, hyperrealistic, authentic skin texture",
    "oil_painting": "oil painting on canvas, thick brushstrokes, classical art, texture",
    "watercolor": "watercolor painting, wet on wet, artistic splashes, pastel colors",
    "pop_art": "pop art style, halftone dots, vibrant colors, comic book aesthetic",
    "fantasy": "fantasy concept art, magical forest, ethereal glow, mystical atmosphere",
    "steampunk": "steampunk aesthetic, brass gears, victorian fashion, steam engine smoke",
    "minimalist": "minimalist photography, clean lines, negative space, soft colors",
}

QUALITY_ENHANCERS = "8k, best quality, masterpiece, ultra highres, photorealistic"
NEGATIVE_PROMPTS = "blur, low quality, distortion, ugly, deformed, text, watermark, bad anatomy, bad hands, cartoon, 3d, illustration"

def get_device_config() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32

def get_depth_estimator():
    global DEPTH_ESTIMATOR
    if DEPTH_ESTIMATOR is not None:
        return DEPTH_ESTIMATOR
    
    try:
        processor = DPTImageProcessor.from_pretrained("Intel/dpt-hybrid-midas")
        model = DPTForDepthEstimation.from_pretrained("Intel/dpt-hybrid-midas")
        DEPTH_ESTIMATOR = (processor, model)
        return DEPTH_ESTIMATOR
    except Exception as e:
        print(f"Failed to load Depth Estimator: {e}")
        return None

def make_canny_condition(image):
    image = np.array(image)
    image = cv2.Canny(image, 100, 200)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    return Image.fromarray(image)

def make_depth_condition(image):
    estimator = get_depth_estimator()
    if estimator is None:
        return image
    
    processor, model = estimator
    
    image_proc = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        output = model(**image_proc)
        prediction = output.predicted_depth
    
    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )
    
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth_image = Image.fromarray(formatted)
    return depth_image.convert("RGB")

def load_ai_model():
    global AI_PIPELINE
    if AI_PIPELINE is not None:
        return AI_PIPELINE

    try:
        print("Loading Ultimate AI System...")
        device, dtype = get_device_config()

        image_encoder = CLIPVisionModelWithProjection.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
            torch_dtype=dtype,
        )
        feature_extractor = CLIPImageProcessor.from_pretrained(
            "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
        )

        controlnets = []
        for model_id in CONTROLNET_IDS:
            cn = ControlNetModel.from_pretrained(model_id, torch_dtype=dtype, use_safetensors=True)
            controlnets.append(cn)

        vae = AutoencoderKL.from_pretrained(VAE_ID, torch_dtype=dtype, use_safetensors=True)

        pipe = StableDiffusionXLControlNetImg2ImgPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=controlnets,
            vae=vae,
            image_encoder=image_encoder,
            feature_extractor=feature_extractor,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16",
            add_watermarker=False,
        )

        print("Loading IP-Adapter...")
        pipe.load_ip_adapter(
            IP_ADAPTER_REPO, 
            subfolder="sdxl_models", 
            weight_name=IP_ADAPTER_FILE
        )
        pipe.set_ip_adapter_scale(0.6)

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="sde-dpmsolver++"
        )

        pipe.vae.enable_tiling()
        pipe.enable_attention_slicing()
        
        if device == "cuda":
            pipe.enable_model_cpu_offload()
            try:
                import xformers
                pipe.enable_xformers_memory_efficient_attention()
            except: pass

        AI_PIPELINE = pipe
        print(f"AI System Ready on {device.upper()}")
        return AI_PIPELINE

    except Exception as e:
        print(f"Failed to load AI model: {e}")
        return None

def smart_resize(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    width, height = image.size
    if width > height:
        new_width = max_size
        new_height = int(height * (max_size / width))
    else:
        new_height = max_size
        new_width = int(width * (max_size / height))
    
    new_width = new_width - (new_width % 8)
    new_height = new_height - (new_height % 8)
    return image.resize((new_width, new_height), Image.LANCZOS)

def build_clean_prompt(user_prompt: str, style: str) -> str:
    style_text = STYLE_PROMPTS.get(style, STYLE_PROMPTS["auto"])
    base_prompt = user_prompt.strip() if user_prompt else ""
    raw_text = f"{base_prompt}, {style_text}, {QUALITY_ENHANCERS}"
    tokens = [t.strip() for t in raw_text.split(',') if t.strip()]
    seen = set()
    unique = []
    for t in tokens:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique.append(t)
    return ", ".join(unique)

def get_dynamic_controlnet_scales(style: str, strength: float) -> List[float]:
    canny_scale = 0.5
    depth_scale = 0.5
    
    line_styles = ["sketch", "anime", "cartoon", "ghibli", "noir"]
    
    if style in line_styles:
        canny_scale = 0.7
        depth_scale = 0.3
    else:
        canny_scale = 0.3
        depth_scale = 0.7
        
    if strength > 0.7:
        canny_scale += 0.2
        depth_scale += 0.2
        
    return [canny_scale, depth_scale]

def generate_ai_image(image_path: str, prompt: str, style: str = "auto", strength: float = 0.35) -> bool:
    try:
        pipe = load_ai_model()
        if pipe is None: return False

        print(f"Processing: {image_path} | Style: {style}")

        original_image = Image.open(image_path).convert("RGB")
        original_image = smart_resize(original_image)

        canny_image = make_canny_condition(original_image)
        depth_image = make_depth_condition(original_image)
        
        control_images = [canny_image, depth_image]
        control_scales = get_dynamic_controlnet_scales(style, strength)
        
        final_prompt = build_clean_prompt(prompt, style)
        
        ip_scale = 0.55 if strength < 0.5 else 0.75
        pipe.set_ip_adapter_scale(ip_scale)

        with torch.inference_mode():
            result = pipe(
                prompt=final_prompt,
                image=original_image,
                control_image=control_images,
                ip_adapter_image=original_image,
                strength=strength,         
                controlnet_conditioning_scale=control_scales,
                guidance_scale=6.5,
                negative_prompt=NEGATIVE_PROMPTS,
                num_inference_steps=DEFAULT_IMAGE_STEPS
            ).images[0]

        result.save(image_path, quality=95, optimize=True)
        print("Image processing completed")
        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

def generate_ai_video(video_path, prompt, style="auto", strength=0.3):
    return False

def compress_video(video_path: str) -> bool:
    try:
        output_path = video_path.replace('.mp4', '_temp.mp4')
        (
            ffmpeg.input(video_path)
            .output(output_path, vcodec='libx264', crf=28, preset='fast', movflags='faststart')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        os.remove(video_path)
        os.rename(output_path, video_path)
        return True
    except: return False

def cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()