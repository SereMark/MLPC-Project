import sys, importlib

def check_package(package_name):
    """Try to import a package and return its version."""
    try:
        module = importlib.import_module(package_name)
        if hasattr(module, '__version__'):
            return f"{package_name}: {module.__version__}"
        else:
            return f"{package_name}: Successfully imported (version unknown)"
    except ImportError:
        return f"{package_name}: Failed to import"

def check_pytorch_cuda():
    """Check if PyTorch can access CUDA."""
    import torch
    
    cuda_available = torch.cuda.is_available()
    cuda_version = torch.version.cuda if cuda_available else "N/A"
    
    print(f"PyTorch CUDA available: {cuda_available}")
    print(f"CUDA version: {cuda_version}")
    
    if cuda_available:
        device_count = torch.cuda.device_count()
        print(f"CUDA device count: {device_count}")
        
        for i in range(device_count):
            device_name = torch.cuda.get_device_name(i)
            print(f"CUDA Device {i}: {device_name}")
        
        try:
            x = torch.rand(5, 5).cuda()
            y = torch.rand(5, 5).cuda()
            z = x @ y
            print("PyTorch CUDA operation: Success")
        except Exception as e:
            print(f"PyTorch CUDA operation: Failed - {str(e)}")
    
    return cuda_available

def test_numpy():
    """Test NumPy functionality."""
    import numpy as np
    
    try:
        arr1 = np.random.rand(1000, 1000)
        arr2 = np.random.rand(1000, 1000)
        result = np.dot(arr1, arr2)
        print("NumPy test: Success")
    except Exception as e:
        print(f"NumPy test: Failed - {str(e)}")

def test_h5py():
    """Test h5py functionality."""
    import h5py
    import os
    import numpy as np
    
    test_file = "test_h5py.h5"
    
    try:
        # Create a test file
        with h5py.File(test_file, 'w') as f:
            data = np.random.rand(100, 100)
            f.create_dataset('test_dataset', data=data)
        
        # Read from the test file
        with h5py.File(test_file, 'r') as f:
            read_data = f['test_dataset'][:]
            if np.array_equal(data, read_data):
                print("h5py test: Success")
            else:
                print("h5py test: Failed - Data mismatch")
                
        # Clean up
        os.remove(test_file)
        
    except Exception as e:
        print(f"h5py test: Failed - {str(e)}")
        if os.path.exists(test_file):
            os.remove(test_file)

def main():
    print("Checking Python version:")
    print(f"Python {sys.version}")
    print("\nChecking required packages:")
    
    # Check all packages
    print(check_package("torch"))
    print(check_package("numpy"))
    print(check_package("h5py"))
    
    print("\nTesting PyTorch CUDA support:")
    check_pytorch_cuda()
    
    print("\nTesting NumPy functionality:")
    test_numpy()
    
    print("\nTesting h5py functionality:")
    test_h5py()
    
    print("\nEnvironment check completed.")

if __name__ == "__main__":
    main()