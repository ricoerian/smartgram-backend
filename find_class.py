import diffusers
import inspect

print("Searching for ControlNetOutput...")

if hasattr(diffusers, 'ControlNetOutput'):
    print("Found in diffusers top level")
elif hasattr(diffusers.models, 'ControlNetOutput'):
    print("Found in diffusers.models")
else:
    print("Not found in top levels. Checking submodules...")
    try:
        from diffusers.models import controlnet
        if hasattr(controlnet, 'ControlNetOutput'):
             print("Found in diffusers.models.controlnet")
    except ImportError:
        print("diffusers.models.controlnet does not exist")

    try:
        from diffusers.pipelines.controlnet import ControlNetOutput
        print("Found in diffusers.pipelines.controlnet")
    except ImportError:
         pass

# List everything in diffusers.models
print("\ndiffusers.models contents:")
print(dir(diffusers.models))
