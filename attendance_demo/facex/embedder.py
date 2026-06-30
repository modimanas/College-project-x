# import cv2
# import numpy as np
# import threading
# from dataclasses import dataclass
# from insightface.app import FaceAnalysis

# @dataclass
# class FaceEmbedding:
#     """
#     Standard container wrapping vector data and integer-mapped 
#     bounding boxes to feed into Matcher and Worker seamlessly.
#     """
#     vector: np.ndarray  # 512-dimensional normalized face descriptor
#     bbox: list          # [x1, y1, x2, y2] bounding box coordinates

# class Embedder:
#     def __init__(self, model_name='antelopev2', use_gpu=False):
#         """
#         Dynamically loads insightface model packs and handles execution contexts.
#         :param model_name: 'antelopev2' (high accuracy) or 'buffalo_l' (balanced)
#         :param use_gpu: True for CUDA (Linux), False uses CoreML on macOS (Apple Silicon)
#         """
#         self.model_name = model_name

#         import platform
#         if use_gpu:
#             providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
#             ctx_id = 0
#             print(f"🚀 Initializing {self.model_name} on GPU (CUDA)...")
#         elif platform.system() == 'Darwin':
#             # CoreML leverages Apple Neural Engine + Metal GPU on M-series chips
#             providers = ['CoreMLExecutionProvider', 'CPUExecutionProvider']
#             ctx_id = -1
#             print(f"🍎 Initializing {self.model_name} on Apple Silicon (CoreML)...")
#         else:
#             providers = ['CPUExecutionProvider']
#             ctx_id = -1
#             print(f"💻 Initializing {self.model_name} on CPU...")

#         self.app = FaceAnalysis(name=self.model_name, providers=providers)
#         self.app.prepare(ctx_id=ctx_id, det_size=(640, 640))
#         self._lock = threading.Lock()   # CoreML EP is not thread-safe for concurrent calls

#     def detect(self, frame):
#         """
#         Executes one-shot face localization and descriptor generation on the target frame.
#         Lock ensures only one thread calls into CoreML/ONNX at a time (not thread-safe).
#         """
#         if frame is None:
#             return []
#         with self._lock:
#             return self.app.get(frame)

#     def embed_face(self, face, frame=None):
#         """
#         Translates raw output objects into structured dataclass payloads.
#         """
#         # Feature array extraction (already pre-calculated during self.app.get)
#         vector = face.normed_embedding
        
#         # Format coordinate boundaries safely as type integer
#         x1, y1, x2, y2 = face.bbox.astype(int)
        
#         # Bound coordinates to image perimeter frames to prevent index errors
#         if frame is not None:
#             h, w, _ = frame.shape
#             x1, y1 = max(0, x1), max(0, y1)
#             x2, y2 = min(w, x2), min(h, y2)

#         return FaceEmbedding(vector=vector, bbox=[x1, y1, x2, y2])


# # =====================================================================
# # CONFIGURATION SWITCHBOARD FOR DEMO.PY
# # =====================================================================
# # Simply swap the variable assignment below to switch models instantly!

# # antelopev2 + CoreML on Apple Silicon (auto-detected)
# default_embedder = Embedder(model_name='antelopev2', use_gpu=False)



import cv2
import numpy as np
import threading
from dataclasses import dataclass
from insightface.app import FaceAnalysis
import onnxruntime as ort


@dataclass
class FaceEmbedding:
    """
    Standard container wrapping vector data and integer-mapped 
    bounding boxes to feed into Matcher and Worker seamlessly.
    """
    vector: np.ndarray  # 512-dimensional normalized face descriptor
    bbox: list          # [x1, y1, x2, y2] bounding box coordinates


class Embedder:
    def __init__(self, model_name='antelopev2', use_gpu=True):
        """
        Dynamically loads insightface model packs and handles execution contexts.
        """

        self.model_name = model_name

        print("======================================")
        print("ONNX Available Providers:")
        print(ort.get_available_providers())
        print("======================================")

        # -------------------------------------------------------
        # GPU MODE (Windows + NVIDIA CUDA)
        # -------------------------------------------------------
        if use_gpu:
            providers = ['CUDAExecutionProvider']
            ctx_id = 0
            print(f"🚀 Initializing {self.model_name} on GPU (CUDA)...")

        # -------------------------------------------------------
        # CPU MODE
        # -------------------------------------------------------
        else:
            providers = ['CPUExecutionProvider']
            ctx_id = -1
            print(f"💻 Initializing {self.model_name} on CPU...")

        try:
            self.app = FaceAnalysis(
                name=self.model_name,
                providers=providers
            )
            print("Requested providers:", providers)
            self.app.prepare(
                ctx_id=ctx_id,
                det_size=(640, 640)
            )

            print("✅ Model Loaded Successfully")
            print("Using Providers:", providers)

        except Exception as e:
            print("❌ GPU initialization failed")
            print("ERROR:", e)

            print("⚠️ Falling back to CPU...")

            self.app = FaceAnalysis(
                name=self.model_name,
                providers=['CPUExecutionProvider']
            )

            self.app.prepare(
                ctx_id=-1,
                det_size=(640, 640)
            )

        # Note: Modern NVIDIA GPUs support concurrent streams
        # Removing lock allows parallel processing on all 3 cameras
        # self._lock = threading.Lock()

    def detect(self, frame):
        """
        Executes one-shot face localization and descriptor generation on the target frame.
        No lock = parallel GPU processing across multiple threads.
        """
        if frame is None:
            return []

        # No lock - allows concurrent GPU calls
        faces = self.app.get(frame)
        return faces

    # def detect(self, frame):

    #  if frame is None:
    #     return []

    #  return self.app.get(frame)

    def embed_face(self, face, frame=None):
        """
        Translates raw output objects into structured dataclass payloads.
        """

        # Feature array extraction
        vector = face.normed_embedding

        # Format coordinate boundaries safely as integer
        x1, y1, x2, y2 = face.bbox.astype(int)

        # Bound coordinates safely
        if frame is not None:
            h, w, _ = frame.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

        return FaceEmbedding(
            vector=vector,
            bbox=[x1, y1, x2, y2]
        )


# =====================================================================
# CONFIGURATION SWITCHBOARD FOR DEMO.PY
# =====================================================================

# Same as your code, only GPU enabled
default_embedder = Embedder(
    model_name='antelopev2',
    use_gpu=True
)