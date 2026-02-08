import os
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

import cv2
import numpy as np
import torch
import gc
import time
from typing import Optional, Tuple, List
from contextlib import contextmanager
from PIL import Image, ImageEnhance, ImageFilter

from diffusers import (
    ControlNetModel,
    StableDiffusionXLControlNetPipeline,
    StableDiffusionXLImg2ImgPipeline,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
    EulerAncestralDiscreteScheduler
)
from compel import Compel, ReturnedEmbeddingsType

try:
    from controlnet_aux import OpenposeDetector
    import mediapipe
    OPENPOSE_AVAILABLE = True
except (ImportError, AttributeError, UserWarning):
    OPENPOSE_AVAILABLE = False

BASE_MODEL_ID = "RunDiffusion/Juggernaut-XL-v9"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
CONTROLNET_ID_CANNY = "diffusers/controlnet-canny-sdxl-1.0"
CONTROLNET_ID_OPENPOSE = "xinsir/controlnet-openpose-sdxl-1.0"

MAX_IMAGE_SIZE = 1024
HIRES_SCALE = 2.0
DEFAULT_INFERENCE_STEPS = 40
DEFAULT_CFG_SCALE = 8.5
DEFAULT_DENOISE_STRENGTH = 0.25

STYLE_PROMPTS = {
    "auto": "masterpiece, best quality, ultra detailed, 8k uhd, dslr, professional photography, soft lighting, high dynamic range, film grain, fujifilm xt3, sharp focus",
    "noir": "film noir aesthetic, monochrome perfection, dramatic chiaroscuro, cinematic lighting, vintage thriller atmosphere, deep shadows, high contrast",
    "sepia": "vintage photograph, authentic sepia tone, 1950s golden era, fine grain texture, nostalgic atmosphere, aged paper quality",
    "sketch": "professional charcoal sketch, detailed graphite drawing, artistic line work, precise shading technique, textured paper, fine art quality",
    "cyber": "cyberpunk metropolis, intense neon illumination, futuristic haute couture, rain-slicked streets, electric blue and hot pink ambiance, blade runner aesthetic",
    "hdr": "professional hdr photography, dramatic atmospheric sky, tack sharp focus, vivid saturated colors, hyperdetailed texture, high dynamic range",
    "cartoon": "pixar disney quality, premium 3d render, adorable character design, vibrant saturated palette, smooth subsurface scattering",
    "anime": "premium anime artwork, makoto shinkai quality, meticulous detail, atmospheric background, volumetric lighting",
    "ghibli": "studio ghibli masterpiece, delicate watercolor background, serene peaceful atmosphere, whimsical enchanting mood, miyazaki style",
    "realistic": "ultra realistic photography, professional dslr, prime 85mm lens, visible skin pores, hyperrealistic details, authentic skin texture, natural lighting",
    "oil_painting": "classical oil painting on canvas, visible thick brushstrokes, renaissance art quality, rich color depth, textured surface",
    "watercolor": "professional watercolor painting, wet on wet technique, artistic color bleeds, soft pastel palette, paper texture visible",
    "pop_art": "andy warhol pop art style, ben-day halftone dots, vibrant bold colors, comic book aesthetic, screen print quality",
    "fantasy": "epic fantasy concept art, enchanted mystical forest, ethereal magical glow, dreamlike atmosphere, professional illustration",
    "steampunk": "victorian steampunk aesthetic, intricate brass gears, elaborate period fashion, atmospheric steam, copper and bronze tones",
    "minimalist": "minimalist fine art photography, clean geometric lines, deliberate negative space, muted sophisticated palette, zen composition",
}

QUALITY_ENHANCERS = "extremely detailed, 8k resolution, best quality possible, masterpiece artwork, ultra high resolution, photorealistic rendering, professional grade, award winning, perfect composition"
NEGATIVE_PROMPTS = "ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers, extra fingers, fused fingers, too many fingers, long neck, mutation, mutated, mutilated, mangled, old, surreal, duplicate, morbid, gross proportions, missing arms, missing legs, extra arms, extra legs, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, lowres, low resolution, jpeg artifacts, signature, watermark, username, artist name, trademark, title, multiple view, reference sheet, error, text, logo, copyright, grainy, overexposed, underexposed, oversaturated, desaturated, amateur, bad proportions, bad shadow, bad highlights, bad lighting, cross-eyed, asymmetric eyes, dehydrated, bad framing, cut off, draft, disfigured"


def get_device_config() -> Tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        return "cuda", torch.float16
    return "cpu", torch.float32


def analyze_image_complexity(image: Image.Image) -> dict:
    img_array = np.array(image.convert("L"))
    
    edges = cv2.Canny(img_array, 30, 120)
    edge_density = np.sum(edges > 0) / edges.size
    
    contrast = np.std(img_array)
    brightness = np.mean(img_array)
    
    laplacian_var = cv2.Laplacian(img_array, cv2.CV_64F).var()
    
    img_rgb = np.array(image)
    color_variance = np.mean([np.std(img_rgb[:,:,i]) for i in range(3)])
    
    blur = cv2.GaussianBlur(img_array, (9, 9), 0)
    detail_level = np.mean(np.abs(img_array.astype(float) - blur.astype(float)))
    
    sobelx = cv2.Sobel(img_array, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_array, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    texture_complexity = np.mean(gradient_magnitude)
    
    hist = cv2.calcHist([img_array], [0], None, [256], [0, 256])
    hist = hist.flatten() / hist.sum()
    entropy = -np.sum(hist * np.log2(hist + 1e-7))
    
    return {
        "edge_density": edge_density,
        "contrast": contrast,
        "brightness": brightness,
        "laplacian_variance": laplacian_var,
        "color_variance": color_variance,
        "detail_level": detail_level,
        "texture_complexity": texture_complexity,
        "entropy": entropy,
        "is_complex": edge_density > 0.08 or texture_complexity > 25,
        "is_low_contrast": contrast < 35,
        "is_dark": brightness < 75,
        "is_bright": brightness > 185,
        "has_fine_details": detail_level > 12,
        "is_colorful": color_variance > 45,
        "is_high_entropy": entropy > 6.5
    }


def compute_optimal_strength(image: Image.Image, style: str) -> float:
    analysis = analyze_image_complexity(image)
    
    base_strength = 0.35
    
    edge_density = analysis["edge_density"]
    if edge_density > 0.25:
        base_strength -= 0.25
    elif edge_density > 0.15:
        base_strength -= 0.15
    elif edge_density > 0.10:
        base_strength -= 0.08
    elif edge_density < 0.04:
        base_strength += 0.18
    elif edge_density < 0.06:
        base_strength += 0.12
    
    sharpness = analysis["laplacian_variance"]
    if sharpness > 1000:
        base_strength -= 0.18
    elif sharpness > 600:
        base_strength -= 0.10
    elif sharpness > 400:
        base_strength -= 0.05
    elif sharpness < 80:
        base_strength += 0.20
    elif sharpness < 150:
        base_strength += 0.12
    
    contrast = analysis["contrast"]
    if contrast > 75:
        base_strength -= 0.10
    elif contrast > 60:
        base_strength -= 0.05
    elif contrast < 25:
        base_strength += 0.18
    elif contrast < 40:
        base_strength += 0.10
    
    detail = analysis["detail_level"]
    if detail > 25:
        base_strength -= 0.12
    elif detail > 18:
        base_strength -= 0.06
    elif detail < 6:
        base_strength += 0.15
    elif detail < 10:
        base_strength += 0.08
    
    color_var = analysis["color_variance"]
    if color_var > 70:
        base_strength -= 0.08
    elif color_var > 55:
        base_strength -= 0.04
    elif color_var < 25:
        base_strength += 0.10
    elif color_var < 35:
        base_strength += 0.06
    
    texture = analysis["texture_complexity"]
    if texture > 35:
        base_strength -= 0.10
    elif texture < 15:
        base_strength += 0.08
    
    entropy = analysis["entropy"]
    if entropy > 7.0:
        base_strength -= 0.08
    elif entropy < 5.5:
        base_strength += 0.10
    
    if analysis["is_dark"]:
        base_strength += 0.10
    elif analysis["is_bright"]:
        base_strength += 0.06
    
    if sharpness > 600 and contrast > 55 and edge_density > 0.12:
        base_strength -= 0.12
    
    if sharpness < 120 and contrast < 35 and edge_density < 0.06:
        base_strength += 0.15
    
    style_modifiers = {
        "realistic": -0.05,
        "sketch": 0.10,
        "watercolor": 0.08,
        "oil_painting": 0.05,
        "cyber": -0.03,
        "hdr": -0.05,
        "fantasy": 0.03
    }
    base_strength += style_modifiers.get(style, 0.0)
    
    final_strength = max(0.15, min(base_strength, 0.75))
    print(f"Intelligent strength computed: {final_strength:.3f} (edge: {edge_density:.3f}, sharpness: {sharpness:.1f}, contrast: {contrast:.1f})")
    return final_strength


def compute_adaptive_canny_thresholds(image: Image.Image) -> Tuple[int, int]:
    gray = np.array(image.convert("L"))
    
    median_val = np.median(gray)
    std_val = np.std(gray)
    
    sigma = 0.33
    low_threshold = int(max(0, (1.0 - sigma) * median_val))
    high_threshold = int(min(255, (1.0 + sigma) * median_val))
    
    if std_val < 30:
        low_threshold = int(low_threshold * 0.8)
        high_threshold = int(high_threshold * 0.8)
    elif std_val > 70:
        low_threshold = int(low_threshold * 1.2)
        high_threshold = int(high_threshold * 1.2)
    
    low_threshold = max(40, min(low_threshold, 120))
    high_threshold = max(80, min(high_threshold, 220))
    
    return low_threshold, high_threshold


def preprocess_image(image: Image.Image) -> Image.Image:
    analysis = analyze_image_complexity(image)
    
    processed = image.copy()
    
    if analysis["is_low_contrast"]:
        enhancer = ImageEnhance.Contrast(processed)
        processed = enhancer.enhance(1.4)
    
    if analysis["is_dark"]:
        enhancer = ImageEnhance.Brightness(processed)
        processed = enhancer.enhance(1.25)
    elif analysis["is_bright"]:
        enhancer = ImageEnhance.Brightness(processed)
        processed = enhancer.enhance(0.88)
    
    if not analysis["has_fine_details"]:
        enhancer = ImageEnhance.Sharpness(processed)
        processed = enhancer.enhance(1.15)
    
    if analysis["is_colorful"]:
        enhancer = ImageEnhance.Color(processed)
        processed = enhancer.enhance(1.05)
    
    return processed


def make_canny_condition(image: Image.Image) -> Image.Image:
    low_thresh, high_thresh = compute_adaptive_canny_thresholds(image)
    
    img_array = np.array(image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    
    edges = cv2.Canny(blurred, low_thresh, high_thresh)
    
    kernel = np.ones((2, 2), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    edges = edges[:, :, None]
    edges = np.concatenate([edges, edges, edges], axis=2)
    
    return Image.fromarray(edges)


def make_openpose_condition(image: Image.Image) -> Image.Image:
    if not OPENPOSE_AVAILABLE:
        raise RuntimeError("OpenPose not available")
    openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    result = openpose_detector(image)
    del openpose_detector
    gc.collect()
    return result


def cleanup_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    time.sleep(0.15)
    gc.collect()
    if torch.cuda.is_available():
        with torch.cuda.device('cuda:0'):
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.reset_max_memory_allocated()
    time.sleep(0.15)
    gc.collect()


def smart_resize(image: Image.Image, max_size: int = MAX_IMAGE_SIZE) -> Image.Image:
    width, height = image.size
    
    aspect_ratio = width / height
    
    if width > height:
        new_width = max_size
        new_height = int(max_size / aspect_ratio)
    else:
        new_height = max_size
        new_width = int(max_size * aspect_ratio)
    
    new_width = (new_width // 64) * 64
    new_height = (new_height // 64) * 64
    
    new_width = max(256, min(new_width, 1024))
    new_height = max(256, min(new_height, 1024))
    
    return image.resize((new_width, new_height), Image.LANCZOS)


def build_smart_prompt(user_prompt: str, style: str, image_analysis: dict) -> str:
    style_text = STYLE_PROMPTS.get(style, STYLE_PROMPTS["auto"])
    
    base_prompt = user_prompt.strip() if user_prompt else "high quality image"
    
    context_hints = []
    
    if image_analysis["is_complex"]:
        context_hints.append("intricate fine details")
    if image_analysis["has_fine_details"]:
        context_hints.append("sharp precise textures")
    if image_analysis["is_dark"]:
        context_hints.append("dramatic moody lighting")
    if image_analysis["is_colorful"]:
        context_hints.append("rich vibrant colors")
    if image_analysis["is_high_entropy"]:
        context_hints.append("complex composition")
    
    context_str = ", ".join(context_hints) if context_hints else ""
    
    if context_str:
        raw_text = f"{base_prompt}, {style_text}, {context_str}, {QUALITY_ENHANCERS}"
    else:
        raw_text = f"{base_prompt}, {style_text}, {QUALITY_ENHANCERS}"
    
    tokens = [t.strip() for t in raw_text.split(',') if t.strip()]
    
    seen = set()
    unique = []
    for t in tokens:
        t_lower = t.lower()
        if t_lower not in seen:
            seen.add(t_lower)
            unique.append(t)
    
    return ", ".join(unique)


@contextmanager
def ai_pipeline_context(use_openpose: bool = False):
    pipe = None
    compel = None
    controlnets = []
    vae = None
    
    try:
        print("Initializing AI pipeline with maximum quality settings...")
        device, dtype = get_device_config()
        
        cleanup_gpu_memory()
        
        controlnet_canny = ControlNetModel.from_pretrained(
            CONTROLNET_ID_CANNY,
            torch_dtype=dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )
        controlnets.append(controlnet_canny)
        
        if use_openpose and OPENPOSE_AVAILABLE:
            controlnet_openpose = ControlNetModel.from_pretrained(
                CONTROLNET_ID_OPENPOSE,
                torch_dtype=dtype,
                use_safetensors=True,
                low_cpu_mem_usage=True
            )
            controlnets.append(controlnet_openpose)

        vae = AutoencoderKL.from_pretrained(
            VAE_ID,
            torch_dtype=dtype,
            use_safetensors=True,
            low_cpu_mem_usage=True
        )

        pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            BASE_MODEL_ID,
            controlnet=controlnets if len(controlnets) > 1 else controlnets[0],
            vae=vae,
            torch_dtype=dtype,
            use_safetensors=True,
            variant="fp16",
            add_watermarker=False,
        )

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            use_karras_sigmas=True,
            algorithm_type="dpmsolver++",
            solver_order=2
        )

        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
        pipe.vae.config.force_upcast = False

        if torch.cuda.is_available():
            pipe.to(device)
            pipe.unet.to(memory_format=torch.channels_last)
            pipe.enable_attention_slicing(slice_size=1)
            pipe.enable_vae_slicing()
            
            compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
                device=device
            )
            try:
                pipe.enable_xformers_memory_efficient_attention()
                print("xFormers memory efficient attention enabled")
            except Exception:
                print("xFormers not available, using standard attention")
        else:
            compel = Compel(
                tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
                text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
                returned_embeddings_type=ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED,
                requires_pooled=[False, True],
            )

        print(f"Pipeline ready | Device: {device} | Maximum quality mode active")
        yield pipe, compel
        
    finally:
        print("Unloading AI pipeline...")
        
        if compel is not None:
            del compel
        
        if pipe is not None:
            components = ['controlnet', 'unet', 'vae', 'text_encoder', 'text_encoder_2']
            for comp_name in components:
                if hasattr(pipe, comp_name):
                    comp = getattr(pipe, comp_name)
                    if comp is not None:
                        if isinstance(comp, list):
                            for item in comp:
                                if item is not None and hasattr(item, 'to'):
                                    item.to('cpu')
                                del item
                        else:
                            if hasattr(comp, 'to'):
                                comp.to('cpu')
                            del comp
                        setattr(pipe, comp_name, None)
            del pipe
        
        for cn in controlnets:
            if cn is not None:
                if hasattr(cn, 'to'):
                    cn.to('cpu')
                del cn
        controlnets.clear()
        
        if vae is not None:
            if hasattr(vae, 'to'):
                vae.to('cpu')
            del vae
        
        cleanup_gpu_memory()
        time.sleep(0.25)
        cleanup_gpu_memory()
        print("Pipeline fully unloaded and memory cleared")


def hires_fix(pipe, image: Image.Image, prompt_embeds, pooled_embeds, negative_embeds, negative_pooled, steps: int, cfg: float, denoise: float) -> Image.Image:
    width, height = image.size
    target_width = int(width * HIRES_SCALE)
    target_height = int(height * HIRES_SCALE)
    
    target_width = (target_width // 64) * 64
    target_height = (target_height // 64) * 64
    
    cleanup_gpu_memory()

    refiner = StableDiffusionXLImg2ImgPipeline(
        vae=pipe.vae,
        text_encoder=pipe.text_encoder,
        text_encoder_2=pipe.text_encoder_2,
        tokenizer=pipe.tokenizer,
        tokenizer_2=pipe.tokenizer_2,
        unet=pipe.unet,
        scheduler=pipe.scheduler,
    )
    
    if torch.cuda.is_available():
        refiner.enable_attention_slicing(slice_size=1)
        refiner.enable_vae_slicing()
        try:
            refiner.enable_xformers_memory_efficient_attention()
        except Exception:
            pass
            
    refiner.vae.config.force_upcast = False
    
    hires_steps = max(steps // 2, 20)
    
    with torch.inference_mode():
        refined = refiner(
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_embeds,
            negative_prompt_embeds=negative_embeds,
            negative_pooled_prompt_embeds=negative_pooled,
            image=image,
            strength=denoise,
            num_inference_steps=hires_steps,
            guidance_scale=cfg,
            width=target_width,
            height=target_height,
        ).images[0]
    
    del refiner
    cleanup_gpu_memory()
    
    return refined


def generate_ai_image(
    image_path: str, 
    prompt: str, 
    style: str = "auto", 
    strength_canny: Optional[float] = None,
    strength_openpose: float = 0.35,
    use_openpose: bool = True,
    use_hires: bool = False,
    inference_steps: int = DEFAULT_INFERENCE_STEPS,
    cfg_scale: float = DEFAULT_CFG_SCALE,
    denoise_strength: float = DEFAULT_DENOISE_STRENGTH,
    strength: Optional[float] = None
) -> bool:
    original_image = None
    preprocessed_image = None
    canny_image = None
    openpose_image = None
    control_images = []
    result = None
    
    try:
        device, dtype = get_device_config()
        
        print(f"Processing: {image_path}")
        print(f"Style: {style} | Steps: {inference_steps} | CFG: {cfg_scale}")
        print(f"Max resolution: {MAX_IMAGE_SIZE}px | Hires: {use_hires}")

        original_image = Image.open(image_path).convert("RGB")
        original_image = smart_resize(original_image, max_size=MAX_IMAGE_SIZE)
        
        preprocessed_image = preprocess_image(original_image)
        image_analysis = analyze_image_complexity(preprocessed_image)
        
        print(f"Image analysis: complexity={image_analysis['is_complex']}, details={image_analysis['has_fine_details']}, entropy={image_analysis['entropy']:.2f}")
        
        if strength_canny is None:
            strength_canny = compute_optimal_strength(preprocessed_image, style)
        else:
            print(f"Using manual ControlNet strength: {strength_canny:.3f}")
        
        canny_image = make_canny_condition(preprocessed_image)
        control_images = [canny_image]
        control_scales = [strength_canny]
        
        if use_openpose and OPENPOSE_AVAILABLE:
            try:
                openpose_image = make_openpose_condition(preprocessed_image)
                control_images.append(openpose_image)
                control_scales.append(strength_openpose)
                print(f"OpenPose ControlNet strength: {strength_openpose:.3f}")
            except Exception as e:
                print(f"OpenPose skipped: {e}")
        elif use_openpose and not OPENPOSE_AVAILABLE:
            print("OpenPose unavailable (dependencies missing)")
        
        final_prompt = build_smart_prompt(prompt, style, image_analysis)
        print(f"Enhanced prompt: {final_prompt[:150]}...")
        
        final_control_scale = control_scales if len(control_scales) > 1 else control_scales[0]

        with ai_pipeline_context(use_openpose=use_openpose and OPENPOSE_AVAILABLE) as (pipe, compel):
            with torch.no_grad():
                conditioning, pooled = compel([final_prompt, NEGATIVE_PROMPTS])
                conditioning = conditioning.to(dtype)
                pooled = pooled.to(dtype)
            
            print("Generating image with maximum quality settings...")
            with torch.inference_mode(), torch.no_grad():
                latents = pipe(
                    prompt_embeds=conditioning[0:1],
                    negative_prompt_embeds=conditioning[1:2],
                    pooled_prompt_embeds=pooled[0:1],
                    negative_pooled_prompt_embeds=pooled[1:2],
                    image=control_images,
                    controlnet_conditioning_scale=final_control_scale,
                    guidance_scale=cfg_scale,
                    num_inference_steps=inference_steps,
                    output_type="latent",
                ).images[0]
                
                del control_images[:]
                del control_scales
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                cleanup_gpu_memory()
                
                print("Decoding latents to image...")
                latents = latents.unsqueeze(0).to(pipe.vae.dtype)
                image = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
                
                del latents
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                cleanup_gpu_memory()
                
                image = (image / 2 + 0.5).clamp(0, 1)
                image = image.cpu().permute(0, 2, 3, 1).float().numpy()
                result = Image.fromarray((image[0] * 255).round().astype("uint8"))
                
                del image
                cleanup_gpu_memory()
            
            if use_hires:
                print(f"Applying hires fix (scale: {HIRES_SCALE}x)...")
                with torch.no_grad():
                    conditioning_hires, pooled_hires = compel([final_prompt, NEGATIVE_PROMPTS])
                    conditioning_hires = conditioning_hires.to(dtype)
                    pooled_hires = pooled_hires.to(dtype)
                
                result = hires_fix(
                    pipe, 
                    result, 
                    conditioning_hires[0:1], 
                    pooled_hires[0:1], 
                    conditioning_hires[1:2], 
                    pooled_hires[1:2], 
                    inference_steps, 
                    cfg_scale, 
                    denoise_strength
                )
                
                conditioning_hires = conditioning_hires.cpu()
                pooled_hires = pooled_hires.cpu()
                del conditioning_hires, pooled_hires
                cleanup_gpu_memory()
            
            conditioning = conditioning.cpu()
            pooled = pooled.cpu()
            del conditioning, pooled
            cleanup_gpu_memory()

        result.save(image_path, quality=98, optimize=True, subsampling=0)
        print("✅ Image generation completed successfully")
        return True

    except Exception as e:
        print(f"❌ Error during generation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        for obj in [original_image, preprocessed_image, canny_image, openpose_image, result]:
            if obj is not None:
                del obj
        
        if control_images:
            control_images.clear()
        
        cleanup_gpu_memory()
        time.sleep(0.2)
        cleanup_gpu_memory()
        print("Memory cleanup completed")


def generate_ai_video(video_path: str, prompt: str, style: str = "auto", strength: float = 0.3) -> bool:
    return False


def compress_video(video_path: str) -> bool:
    try:
        import ffmpeg
        output_path = video_path.replace('.mp4', '_temp.mp4')
        (
            ffmpeg.input(video_path)
            .output(output_path, vcodec='libx264', crf=23, preset='slow', movflags='faststart')
            .overwrite_output()
            .run(capture_stdout=True, capture_stderr=True)
        )
        os.remove(video_path)
        os.rename(output_path, video_path)
        return True
    except Exception:
        return False