# test_gpu.py — Run this to verify everything works
import torch
import transformers
import sentence_transformers
import lightgbm
import sklearn
import pandas
import numpy

print("=" * 50)
print("SYSTEM CHECK")
print("=" * 50)
print(f"PyTorch:            {torch.__version__}")
print(f"Transformers:       {transformers.__version__}")
print(f"Sentence-Transform: {sentence_transformers.__version__}")
print(f"Scikit-learn:       {sklearn.__version__}")
print(f"CUDA Available:     {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"GPU Name:           {torch.cuda.get_device_name(0)}")
    vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"VRAM Total:         {vram:.1f} GB")
    print(f"\n✅ GPU is READY for training!")
else:
    print("\n❌ GPU not detected. Check CUDA installation.")

# Quick GPU test
x = torch.randn(1000, 1000).cuda()
y = torch.randn(1000, 1000).cuda()
z = torch.mm(x, y)
print(f"\nGPU Matrix multiply test: PASSED ✅")
print(f"Result shape: {z.shape}")