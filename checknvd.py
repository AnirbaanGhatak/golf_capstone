import torch
import torchvision

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
cudnn_available = torch.backends.cudnn.is_available()

def test_gpu_usage():
    if not cuda_available:
        print("CUDA is not available!")
        return
    if not cudnn_available:
        print("cuDNN is not available!")
        return
    
    print("Torch version:", torch.__version__)
    print("Torchvision version:", torchvision.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Torch CUDA version:", torch.version.cuda)

    
    print("CUDA and cuDNN are available!")
    print("Running a simple tensor operation to spike GPU usage...")
    
    device = torch.device("cuda")
    x = torch.randn((10000, 10000), device=device)  # Large tensor allocation
    y = torch.matmul(x, x)  # Heavy computation
    print("Computation complete!")
    
    del x, y
    torch.cuda.empty_cache()
    
if __name__ == "__main__":
    test_gpu_usage()

