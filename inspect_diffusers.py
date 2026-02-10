
import inspect
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

print("\n=== MultiControlNetModel.forward args ===")
sig = inspect.signature(MultiControlNetModel.forward)
print(sig)

try:
    print("\n=== MultiControlNetModel.forward source (first 40 lines) ===")
    source = inspect.getsource(MultiControlNetModel.forward)
    print("\n".join(source.splitlines()[:40]))
except Exception:
    pass
