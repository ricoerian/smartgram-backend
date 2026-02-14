import cv2
import numpy as np
from PIL import Image
from typing import Union
import os
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.utils import face_align

class FaceRestorer:
    def __init__(self, model_path: str = "weights/codeformer.onnx", fidelity = 0.9, providers=['CUDAExecutionProvider', 'CPUExecutionProvider']):
        self.model_path = model_path
        self.fidelity = fidelity
        self.session = None
        self.app = None
        self.providers = providers
        self._initialize()

    def _initialize(self):
        """
        Initialize ONNX runtime session for GFPGAN and FaceAnalysis for detection.
        """
        # 1. Initialize Face Detection (reused from checks if possible, but here meaningful)
        try:
            # We use the same 'buffalo_l' model pack as FaceSwapper
            self.app = FaceAnalysis(name='buffalo_l', providers=self.providers)
            self.app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as e:
            print(f"Warning: Failed to init FaceAnalysis for restoration: {e}")
            self.app = None

        # 2. Initialize GFPGAN ONNX
        if not os.path.exists(self.model_path):
            print(f"Warning: GFPGAN ONNX model not found at {self.model_path}")
            return

        try:
            self.session = ort.InferenceSession(self.model_path, providers=self.providers)
            # print("GFPGAN ONNX initialized successfully.")
        except Exception as e:
            print(f"Failed to initialize GFPGAN ONNX: {e}")
            try:
                print("Retrying GFPGAN ONNX with CPU...")
                self.session = ort.InferenceSession(self.model_path, providers=['CPUExecutionProvider'])
                print("GFPGAN ONNX initialized on CPU.")
            except Exception as e2:
                 print(f"Failed to initialize GFPGAN ONNX on CPU: {e2}")

    def restore_image(self, image: Union[Image.Image, np.ndarray], fidelity: float = None) -> Image.Image:
        """
        Apply face restoration using ONNX model with InsightFace alignment.
        """
        w = fidelity if fidelity is not None else self.fidelity
        # Convert to BGR numpy
        if isinstance(image, Image.Image):
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        else:
            img_bgr = image.copy()

        if self.session is None or self.app is None:
            # Fallback if models failed to load
            return self._fallback_restore(img_bgr)

        try:
            # 1. Detect faces
            faces = self.app.get(img_bgr)
            if not faces:
                # No faces to restore
                return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))

            # 2. Process each face
            for face in faces:
                # Align face (crop)
                # We use the standard 5 points (kps) for alignment
                # usually GFPGAN expects 512x512 alignment
                aimg = face_align.norm_crop(img_bgr, landmark=face.kps, image_size=512)
                
                # Preprocess for GFPGAN
                # BGR -> RGB -> [0,1] -> Normalize -> CHW -> Batch
                blob = cv2.cvtColor(aimg, cv2.COLOR_BGR2RGB)
                blob = blob.astype(np.float32) / 255.0
                blob = (blob - 0.5) / 0.5
                blob = blob.transpose(2, 0, 1)[np.newaxis, ...]

                # Inference
                input_name = self.session.get_inputs()[0].name
                output = self.session.run(None, {input_name: blob})[0]

                # Postprocess
                output = output.squeeze().transpose(1, 2, 0)
                output = (output * 0.5 + 0.5) * 255.0
                output = output.clip(0, 255).astype(np.uint8)
                output_bgr = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

                # Paste back
                # Inverse transform logic
                # Standard ArcFace 112x112 reference points
                arcface_src = np.array([
                    [38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.7299, 92.2041] ], dtype=np.float32)
                
                # We need to scale these to 512x512 if we want to estimate the transform for 512 directly
                # Or we can just use the transform we got from face_align.norm_crop if we can retrieve it?
                # face_align.norm_crop returns just the image.
                # So we must re-estimate.
                
                # Target points for 512x512
                # Arcface is 112x112.
                # Ratio is 512/112 = 4.5714
                dst = arcface_src * (512 / 112.0)
                
                # Estimate the transform matrix from Original Face Landmarks -> 512x512 Target
                tform = cv2.estimateAffinePartial2D(face.kps, dst, method=cv2.LMEDS)[0]
                
                # Color Transfer
                # Match the restored face color to the original face crop
                # original face crop is `aimg` (aligned image 512x512)
                # output_bgr is the restored face 512x512
                # But aimg is warped. The real target is in img_bgr.
                # However, comparing `output_bgr` to `aimg` is a good proxy because `aimg` is derived from `img_bgr`.
                try:
                    # Resize aimg to 512x512 match if needed (it is 512x512)
                    output_bgr = self._color_transfer(output_bgr, cv2.resize(aimg, (512, 512)))
                except Exception as e_color:
                    print(f"Color transfer failed: {e_color}")
                    pass
                
                try:
                    # Inverse warp
                    inverse_tform = cv2.invertAffineTransform(tform)
                    
                    # Create precise mask
                    mask = np.zeros((512, 512), dtype=np.uint8)
                    
                    # Strategy: Use Convex Hull of landmarks
                    # If 106 landmarks are available, use them.
                    # If not, use the 5 KPS points but expanded significantly.
                    
                    points = None
                    
                    if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
                        # Use 106 landmarks
                        lmks = face.landmark_2d_106
                        ones = np.ones((lmks.shape[0], 1))
                        lmks_homo = np.hstack([lmks, ones])
                        lmks_trans = np.dot(tform, lmks_homo.T).T
                        points = lmks_trans.astype(np.int32)
                    else:
                        # Fallback to 5 KPS
                        # Transform KPS to 512x512
                        kps = face.kps
                        ones = np.ones((kps.shape[0], 1))
                        kps_homo = np.hstack([kps, ones])
                        kps_trans = np.dot(tform, kps_homo.T).T
                        
                        # Expand 5 points to cover forehead and chin roughly
                        # 5 points are: Left Eye, Right Eye, Nose, Left Mouth, Right Mouth
                        # We need to synthesize a forehead point and chin point
                        eye_center = (kps_trans[0] + kps_trans[1]) / 2
                        mouth_center = (kps_trans[3] + kps_trans[4]) / 2
                        nose = kps_trans[2]
                        
                        face_height = np.linalg.norm(eye_center - mouth_center)
                        
                        forehead = eye_center - (mouth_center - eye_center) * 0.8
                        chin = mouth_center + (mouth_center - eye_center) * 0.5
                        jaw_left = kps_trans[3] - (kps_trans[4] - kps_trans[3]) * 0.5
                        jaw_right = kps_trans[4] + (kps_trans[4] - kps_trans[3]) * 0.5
                        
                        points = np.vstack([kps_trans, forehead, chin, jaw_left, jaw_right]).astype(np.int32)

                    if points is not None:
                        hull = cv2.convexHull(points)
                        cv2.fillConvexPoly(mask, hull, 255)
                        
                        # Erode/Dilate to smooth shape?
                        # Actually, we want to expand it slightly to include skin, then soft fade
                        # But GFPGAN output is usually good only on the face.
                        # Let's simple fill convex hull.
                        pass
                    
                    # CRITICAL: Fade out the edges of the 512x512 box
                    # Even with a mask, if the mask touches the edge, it creates a line.
                    # We create a radial gradient or simple border fade.
                    h, w = mask.shape
                    # Zero out a border of 20 pixels
                    cv2.rectangle(mask, (0, 0), (w, 20), 0, -1)
                    cv2.rectangle(mask, (0, h-20), (w, h), 0, -1)
                    cv2.rectangle(mask, (0, 0), (20, h), 0, -1)
                    cv2.rectangle(mask, (w-20, 0), (w, h), 0, -1)
                    
                    # Soften the edges of the mask
                    # mask = cv2.GaussianBlur(mask, (61, 61), 0)
                    # Instead of large blur on hull, let's blur for alpha blend only.
                    
                    # Create binary mask for seamless clone (255 where active)
                    mask_uint8 = (mask > 0).astype(np.uint8) * 255
                    
                    # Warp the enhanced face back
                    warped_face = cv2.warpAffine(output_bgr, inverse_tform, (img_bgr.shape[1], img_bgr.shape[0]))
                    warped_mask = cv2.warpAffine(mask, inverse_tform, (img_bgr.shape[1], img_bgr.shape[0]))
                    
                    # For seamlessClone, we need a center point and a mask.
                    mh, mw = warped_mask.shape[:2]
                    coords = cv2.findNonZero((warped_mask * 255).astype(np.uint8))
                    if coords is not None:
                        x, y, w, h = cv2.boundingRect(coords)
                        center = (int(x + w/2), int(y + h/2))
                        
                        warped_mask_uint8 = (warped_mask * 255).astype(np.uint8)
                        
                        try:
                            # NORMAL_CLONE tries to preserve gradients of src but match destination.
                            # If src has very different structure at edge, it fails.
                            # If src has hard edge at boundary, it propagates.
                            img_bgr = cv2.seamlessClone(warped_face, img_bgr, warped_mask_uint8, center, cv2.NORMAL_CLONE)
                        except Exception as e_clone:
                            print(f"Seamless clone failed: {e_clone}. Fallback to alpha blend.")
                            # Fallback to alpha blend
                            # Re-blur mask for soft blend
                            warped_mask_soft = cv2.GaussianBlur(warped_mask, (51, 51), 0)
                            if len(warped_mask_soft.shape) == 2:
                                warped_mask_soft = warped_mask_soft[..., np.newaxis]
                                
                            img_bgr = img_bgr * (1.0 - warped_mask_soft) + warped_face * warped_mask_soft
                            img_bgr = img_bgr.astype(np.uint8)
                    else:
                         pass
                except Exception as e:
                    print(f"Failed to paste face back: {e}")
                    pass

            return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"GFPGAN restoration loop failed: {e}")
            return self._fallback_restore(img_bgr)
            
    def _color_transfer(self, source, target):
        """
        Transfer color distribution from target to source.
        Simple Mean/Std deviation transfer in LAB color space.
        """
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
        
        source_mean, source_std = cv2.meanStdDev(source_lab)
        target_mean, target_std = cv2.meanStdDev(target_lab)
        
        source_mean = source_mean.flatten()
        source_std = source_std.flatten()
        target_mean = target_mean.flatten()
        target_std = target_std.flatten()
        
        # Avoid division by zero
        source_std[source_std == 0] = 1.0
        
        # Transfer
        # re-centered = (pixel - mean_src) * (std_tgt / std_src) + mean_tgt
        channels = cv2.split(source_lab)
        for i in range(3):
            channels[i] = (channels[i] - source_mean[i]) * (target_std[i] / source_std[i]) + target_mean[i]
            
        result_lab = cv2.merge(channels)
        result_lab = np.clip(result_lab, 0, 255).astype(np.uint8)
        
        return cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)

    def _fallback_restore(self, img_bgr):
        # Fallback enhancement
        try:
             enhanced = cv2.detailEnhance(img_bgr, sigma_s=10, sigma_r=0.15)
             return Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        except Exception:
             return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))