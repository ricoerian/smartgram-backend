from diffusers import ControlNetModel
from diffusers.models.controlnets import ControlNetUnionModel
import inspect

print("=== Loading xinsir/controlnet-union-sdxl-1.0 (config only) ===")
# We won't download the whole model, just check config/cls via from_pretrained if possible, 
# but downloading is slow. We can just check the class of ControlNetUnionModel.

print("\n=== ControlNetUnionModel.forward signature ===")
try:
    sig = inspect.signature(ControlNetUnionModel.forward)
    print(sig)
except Exception as e:
    print(e)
    
print("\n=== Check if ControlNetModel.from_pretrained loads Union ===")
# Ideally we would check this, but without downloading.
# We can check if 'xinsir' is mapped in library or if we know it's a Union model.
# The previous test run output might have hints if I printed type.
# But I wrapped it in UnionMultiControlNetModel.
# Inside UnionMultiControlNetModel generic logic, I assumed standard ControlNetModel.

print("\n=== Pipeline call signature check ===")
from diffusers import StableDiffusionXLControlNetPipeline
# We want to see the source code of the loop where it calls controlnet.
# We can't easily see source processing, but we can check the 'check_inputs' to see if it warns about things.
