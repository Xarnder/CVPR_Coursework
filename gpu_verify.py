import torch

def verify():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print("-" * 30)

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        print(f"✅ Success! Found {device_count} CUDA-enabled GPU(s).")
        for i in range(device_count):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"     VRAM: {vram:.2f} GB")
    else:
        print("❌ Failed! PyTorch cannot see the GPU.")
        print("   Make sure you ran the container with the '--gpus all' flag.")

if __name__ == "__main__":
    verify()