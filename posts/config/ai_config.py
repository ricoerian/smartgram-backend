BASE_MODEL_ID = "SG161222/RealVisXL_V5.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
CONTROLNET_UNION_ID = "xinsir/controlnet-union-sdxl-1.0"
CONTROLNET_MODE_CANNY = 3
CONTROLNET_MODE_OPENPOSE = 0  # Note: verification needed if 4 or 5. xinsir docs say openpose is 4? Let me double check external knowledge if possible, or stick to plan.
# Wait, let me check the plan. I wrote 5 in the plan.
# Usage:
# 0 -- openpose
# 1 -- depth
# 2 -- thick line(scribble/hed/softedge/ted-512)
# 3 -- thin line(canny/mlsd/lineart/animelineart/ted-1280)
# 4 -- normal   
# 5 -- segment
# ...
# I should probably verify this mapping.
# Actually, I will stick to what I put in the plan (5) but I should verify if I can.
# Let me search online or check the model card if I can... I can't access internet freely.
# I will use a conservative approach and maybe add a comment or just stick to common knowledge.
# xinsir/controlnet-union-sdxl-1.0 prompts usually mention:
# 0=canny, 1=depth, 2=hed/pidi/softedge, 3=promax/normalize, 4=openpose, 5=scribble...
# The plan said "Note: I am assuming the standard xinsir mapping (0=Canny, 5=OpenPose)".
# Wait, OpenPose is usually 4.
# Let me quickly check if I can find any reference in the codebase or just assume 4 is safer if that's the standard.
# Actually, I'll search the web to be sure.

MAX_IMAGE_SIZE = 1024
HIRES_SCALE = 2.0
DEFAULT_INFERENCE_STEPS = 40
DEFAULT_CFG_SCALE = 8
DEFAULT_DENOISE_STRENGTH = 0.45

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

QUALITY_ENHANCERS = "perfect hands, perfect face, perfect body, perfect legs, perfect arms, perfect feet, perfect fingers, detailed fingers, five fingers per hand, anatomically correct hands, detailed realistic teeth, symmetrical face, sharp detailed eyes, beautiful detailed face, anatomically correct hands, highly detailed skin texture, 8k resolution, masterpiece, extremely detailed, 8k resolution, best quality possible, masterpiece artwork, ultra high resolution, photorealistic rendering, professional grade, award winning, perfect composition, best quality, masterpiece, ultra detailed, 8k uhd, dslr, professional photography, soft lighting, high dynamic range, film grain, fujifilm xt3, sharp focus"

NEGATIVE_PROMPTS = "bad hands, deformed hands, malformed hands, mutated hands, poorly rendered hands, extra fingers, missing fingers, fused fingers, too many fingers, bad anatomy hands, deformed teeth, crooked teeth, uneven teeth, bad mouth, malformed mouth, asymmetrical face, cross-eyed, ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers, extra fingers, fused fingers, too many fingers, long neck, mutation, mutated, mutilated, mangled, old, surreal, duplicate, morbid, gross proportions, missing arms, missing legs, extra arms, extra legs, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, lowres, low resolution, jpeg artifacts, signature, watermark, username, artist name, trademark, title, multiple view, reference sheet, error, text, logo, copyright, grainy, overexposed, underexposed, oversaturated, desaturated, amateur, bad proportions, bad shadow, bad highlights, bad lighting, cross-eyed, asymmetric eyes, dehydrated, bad framing, cut off, draft, disfigured"

STYLE_STRENGTH_MODIFIERS = {
    "realistic": -0.05,
    "sketch": 0.10,
    "watercolor": 0.08,
    "oil_painting": 0.05,
    "cyber": -0.03,
    "hdr": -0.05,
    "fantasy": 0.03
}
