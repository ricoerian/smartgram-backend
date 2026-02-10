import inspect
try:
    from diffusers.models.controlnets import MultiControlNetUnionModel
    print("Found MultiControlNetUnionModel")
    print("\n=== MultiControlNetUnionModel.forward args ===")
    sig = inspect.signature(MultiControlNetUnionModel.forward)
    print(sig)
    
    print("\n=== MultiControlNetUnionModel.forward source (first 40 lines) ===")
    source = inspect.getsource(MultiControlNetUnionModel.forward)
    print("\n".join(source.splitlines()[:40]))
except ImportError:
    print("MultiControlNetUnionModel not found")
except Exception as e:
    print(f"Error: {e}")
