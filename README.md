# mnist
cnn, resnet, densenet implementation on mnist

## Python Version
- `cnn.py` - CNN model training
- `resnet.py` - ResNet18 model training
- `densenet.py` - DenseNet121 model training
- `test.py` - Test all trained models

## C++/CUDA Version

### Files
- `resnet.cpp` - ResNet18 C++/CUDA implementation using LibTorch
- `CMakeLists.txt` - CMake build configuration
- `Makefile` - Build automation (downloads LibTorch and compiles)

### Requirements
- C++17 compatible compiler (g++ 13.3.0 or later)
- CMake 3.18 or later
- wget and unzip utilities
- CUDA Toolkit (optional, for GPU acceleration)

### Building the C++ version

```bash
make
```

This will automatically:
1. Download LibTorch (CPU or CUDA version based on CUDA availability)
2. Extract and setup LibTorch
3. Configure CMake with LibTorch
4. Build the C++ ResNet implementation
5. Create the `resnet.out` executable

The first build will take some time as it downloads ~163MB for LibTorch.

### Preparing the data

Before running, ensure the MNIST dataset is available:

```bash
# Using Python to download MNIST
python3 -c "from torchvision import datasets; datasets.MNIST(root='./data', train=True, download=True)"
cd data && ln -sf MNIST/raw/* . && cd ..
```

Or simply run `./resnet.out` - it will show an error if data is missing.

### Running the C++ version

```bash
./resnet.out
```

Expected output:
```
Starting ResNet18 training on MNIST...
Training on CPU.
[1,    30] loss: 0.4159
[1,    60] loss: 0.1265
...
[10,   210] loss: 0.0050
Finished Training and Saving the model
```

This will:
- Train a ResNet18 model on MNIST for 10 epochs
- Use batch size of 256
- Use SGD optimizer (lr=0.01, momentum=0.9)
- Print loss every 30 batches
- Save the trained model to `mnist_resnet18.pt`

Training time:
- CPU: ~2-5 minutes
- GPU: ~30-60 seconds

### Cleaning

```bash
make clean      # Remove build artifacts and resnet.out
make distclean  # Remove everything including LibTorch (~163MB)
```

### Implementation Details

The C++ implementation mirrors the Python version:
- ResNet18 architecture with BasicBlock
- Modified first convolution layer (1 channel for grayscale)
- 10 output classes for MNIST digits
- Images resized from 28x28 to 32x32 using bilinear interpolation
- Cross-entropy loss
- SGD optimizer with momentum
