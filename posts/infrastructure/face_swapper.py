import os
import cv2
import numpy as np
import insightface
from insightface.app import FaceAnalysis
import onnxruntime
from typing import Optional, List, Union
from PIL import Image

class FaceSwapper:
    def __init__(self, model_path: str = "weights/inswapper_128.onnx", providers: List[str] = ['CUDAExecutionProvider', 'CPUExecutionProvider']):
        self.model_path = model_path
        self.providers = providers
        self.app = None
        self.swapper = None
        self._initialize()

    def _initialize(self):
        try:
            # Initialize FaceAnalysis for face detection
            self.app = FaceAnalysis(name='buffalo_l', providers=self.providers)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
            
            # Initialize Swapper
            if os.path.exists(self.model_path):
                try:
                    self.swapper = insightface.model_zoo.get_model(self.model_path, providers=self.providers)
                except Exception as e:
                    print(f"Warning: Failed to load Face Swap model with default providers: {e}")
                    print("Retrying with CPUExecutionProvider...")
                    self.swapper = insightface.model_zoo.get_model(self.model_path, providers=['CPUExecutionProvider'])
            else:
                print(f"Warning: Face swap model not found at {self.model_path}")
                # Try to download or handle missing model
                # For now, we assume it will be there or downloaded by user/script
        except Exception as e:
            print(f"Failed to initialize FaceSwapper: {e}")
            self.app = None
            self.swapper = None

    def _scale_landmarks(self, kps, scale=1.0, shift_y=0.0):
        if scale == 1.0 and shift_y == 0.0:
            return kps
            
        # Calculate center of landmarks
        center = np.mean(kps, axis=0)
        
        # Center the landmarks
        centered = kps - center
        
        # Scale
        scaled = centered * scale
        
        # Shift Y (relative to face size)
        # Approximate face height
        face_height = np.linalg.norm(kps[0] - kps[3]) # Eye to mouth roughly
        scaled[:, 1] += (shift_y * face_height)
        
        # Move back
        return scaled + center

    def process_image(self, target_img: Union[Image.Image, np.ndarray], source_img: Union[Image.Image, np.ndarray], scale: float = 1.0, shift_y: float = 0.0) -> Image.Image:
        if self.app is None or self.swapper is None:
            print("FaceSwapper not initialized, skipping.")
            return target_img

        # Convert to numpy/BGR if needed
        if isinstance(target_img, Image.Image):
            target_bgr = cv2.cvtColor(np.array(target_img), cv2.COLOR_RGB2BGR)
        else:
            target_bgr = target_img.copy()

        if isinstance(source_img, Image.Image):
            source_bgr = cv2.cvtColor(np.array(source_img), cv2.COLOR_RGB2BGR)
        else:
            source_bgr = source_img.copy()

        # Detect faces
        source_faces = self.app.get(source_bgr)
        if not source_faces:
            print("No face detected in source image.")
            return target_img
            
        # Sort by size (largest face is likely the main subject)
        source_faces = sorted(source_faces, key = lambda x : x.bbox[2]*x.bbox[3], reverse=True)
        source_face = source_faces[0]

        target_faces = self.app.get(target_bgr)
        if not target_faces:
             print("No face detected in target image.")
             return target_img

        # Swap all faces in target
        res = target_bgr.copy()
        for face in target_faces:
            # Apply scaling/shifting to landmarks for this swap
            if scale != 1.0 or shift_y != 0.0:
                original_kps = face.kps.copy()
                face.kps = self._scale_landmarks(original_kps, scale, shift_y)
                
            res = self.swapper.get(res, face, source_face, paste_back=True)
            
            # Restore original kps if object is reused (though we re-detect usually)
            if scale != 1.0 or shift_y != 0.0:
                 face.kps = original_kps

        return Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB))
