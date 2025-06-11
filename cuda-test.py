import torch

# Check if CUDA (GPU support) is available for PyTorch
is_available = torch.cuda.is_available()

print(f"Is PyTorch able to use the GPU? -> {is_available}")

if is_available:
    # Get the number of GPUs available
    gpu_count = torch.cuda.device_count()
    print(f"Number of available GPUs: {gpu_count}")

    # Get the name of the current GPU
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Name: {gpu_name}")
else:
    print("\nPyTorch cannot find your GPU.")
    print("This is why your training is slow.")
    print("Please follow the steps below to fix this.")