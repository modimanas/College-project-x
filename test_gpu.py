#!/usr/bin/env python
import onnxruntime as ort
import sys

print("=" * 60)
print("ONNX Runtime GPU Verification")
print("=" * 60)
print(f"Python: {sys.version}")
print(f"ONNX Runtime Version: {ort.__version__}")
print(f"\nAvailable Providers: {ort.get_available_providers()}")
print("=" * 60)

# Check if GPU provider is available
if 'CUDAExecutionProvider' in ort.get_available_providers():
    print("✅ GPU (CUDA) is AVAILABLE")
else:
    print("❌ GPU (CUDA) is NOT available - using CPU only")
