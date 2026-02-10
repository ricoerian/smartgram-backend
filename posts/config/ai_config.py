BASE_MODEL_ID = "SG161222/RealVisXL_V5.0"
VAE_ID = "madebyollin/sdxl-vae-fp16-fix"
CONTROLNET_ID_CANNY = "diffusers/controlnet-canny-sdxl-1.0"
CONTROLNET_ID_OPENPOSE = "xinsir/controlnet-openpose-sdxl-1.0"

MAX_IMAGE_SIZE = 1024
HIRES_SCALE = 2.0
DEFAULT_INFERENCE_STEPS = 40
DEFAULT_CFG_SCALE = 9
DEFAULT_DENOISE_STRENGTH = 0.55

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

QUALITY_ENHANCERS = "anatomically correct hands, perfect fingers, highly detailed skin texture, 8k resolution, masterpiece, extremely detailed, 8k resolution, best quality possible, masterpiece artwork, ultra high resolution, photorealistic rendering, professional grade, award winning, perfect composition"

NEGATIVE_PROMPTS = "ugly, deformed, noisy, blurry, distorted, out of focus, bad anatomy, extra limbs, poorly drawn face, poorly drawn hands, missing fingers, extra fingers, fused fingers, too many fingers, long neck, mutation, mutated, mutilated, mangled, old, surreal, duplicate, morbid, gross proportions, missing arms, missing legs, extra arms, extra legs, extra digit, fewer digits, cropped, worst quality, low quality, normal quality, lowres, low resolution, jpeg artifacts, signature, watermark, username, artist name, trademark, title, multiple view, reference sheet, error, text, logo, copyright, grainy, overexposed, underexposed, oversaturated, desaturated, amateur, bad proportions, bad shadow, bad highlights, bad lighting, cross-eyed, asymmetric eyes, dehydrated, bad framing, cut off, draft, disfigured"

STYLE_STRENGTH_MODIFIERS = {
    "realistic": -0.05,
    "sketch": 0.10,
    "watercolor": 0.08,
    "oil_painting": 0.05,
    "cyber": -0.03,
    "hdr": -0.05,
    "fantasy": 0.03
}
