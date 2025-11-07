#include <torch/torch.h>
#include <iostream>
#include <iomanip>

struct ResNet18Impl : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::Sequential layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
    torch::nn::Linear fc{nullptr};

    ResNet18Impl() {
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, 64, 7).stride(2).padding(3).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
        
        layer1 = register_module("layer1", make_layer(64, 64, 2, 1));
        layer2 = register_module("layer2", make_layer(64, 128, 2, 2));
        layer3 = register_module("layer3", make_layer(128, 256, 2, 2));
        layer4 = register_module("layer4", make_layer(256, 512, 2, 2));
        
        fc = register_module("fc", torch::nn::Linear(512, 10));
    }

    torch::nn::Sequential make_layer(int64_t in_channels, int64_t out_channels, 
                                      int64_t num_blocks, int64_t stride) {
        torch::nn::Sequential layers;
        layers->push_back(BasicBlock(in_channels, out_channels, stride, 
                                     in_channels != out_channels));
        for (int i = 1; i < num_blocks; ++i) {
            layers->push_back(BasicBlock(out_channels, out_channels, 1, false));
        }
        return layers;
    }

    struct BasicBlockImpl : torch::nn::Module {
        torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
        torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
        torch::nn::Sequential downsample{nullptr};
        bool use_downsample;

        BasicBlockImpl(int64_t in_channels, int64_t out_channels, 
                       int64_t stride, bool downsample_flag) 
            : use_downsample(downsample_flag) {
            conv1 = register_module("conv1", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                    .stride(stride).padding(1).bias(false)));
            bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
            conv2 = register_module("conv2", torch::nn::Conv2d(
                torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                    .stride(1).padding(1).bias(false)));
            bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
            
            if (use_downsample) {
                downsample = register_module("downsample", torch::nn::Sequential(
                    torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1)
                        .stride(stride).bias(false)),
                    torch::nn::BatchNorm2d(out_channels)
                ));
            }
        }

        torch::Tensor forward(torch::Tensor x) {
            auto identity = x;
            
            auto out = conv1->forward(x);
            out = bn1->forward(out);
            out = torch::relu(out);
            
            out = conv2->forward(out);
            out = bn2->forward(out);
            
            if (use_downsample) {
                identity = downsample->forward(x);
            }
            
            out += identity;
            out = torch::relu(out);
            
            return out;
        }
    };

    using BasicBlock = torch::nn::ModuleHolder<BasicBlockImpl>;

    torch::Tensor forward(torch::Tensor x) {
        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, 3, 2, 1);
        
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        
        x = torch::adaptive_avg_pool2d(x, {1, 1});
        x = x.view({x.size(0), -1});
        x = fc->forward(x);
        
        return x;
    }
};

TORCH_MODULE(ResNet18);

int main() {
    std::cout << "Starting ResNet18 training on MNIST..." << std::endl;
    
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
        device = torch::Device(torch::kCUDA);
    } else {
        std::cout << "Training on CPU." << std::endl;
    }
    
    const int64_t batch_size = 256;
    const int64_t num_epochs = 10;
    const double learning_rate = 0.01;
    const double momentum = 0.9;
    
    auto train_dataset = torch::data::datasets::MNIST("./data")
        .map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
        .map(torch::data::transforms::Stack<>());
    
    auto train_loader = torch::data::make_data_loader(
        std::move(train_dataset),
        torch::data::DataLoaderOptions().batch_size(batch_size).workers(2));
    
    ResNet18 model;
    model->to(device);
    
    torch::optim::SGD optimizer(
        model->parameters(),
        torch::optim::SGDOptions(learning_rate).momentum(momentum));
    
    model->train();
    
    for (int64_t epoch = 1; epoch <= num_epochs; ++epoch) {
        double running_loss = 0.0;
        int batch_idx = 0;
        
        for (auto& batch : *train_loader) {
            auto data = batch.data.to(device);
            auto target = batch.target.to(device);
            
            data = torch::nn::functional::interpolate(data, 
                torch::nn::functional::InterpolateFuncOptions()
                    .size(std::vector<int64_t>{32, 32})
                    .mode(torch::kBilinear)
                    .align_corners(false));
            
            optimizer.zero_grad();
            
            auto output = model->forward(data);
            auto loss = torch::cross_entropy_loss(output, target);
            
            loss.backward();
            optimizer.step();
            
            running_loss += loss.item<double>();
            
            if ((batch_idx + 1) % 30 == 0) {
                std::cout << "[" << epoch << ", " << std::setw(5) << (batch_idx + 1)
                          << "] loss: " << std::fixed << std::setprecision(4)
                          << (running_loss / 30.0) << std::endl;
                running_loss = 0.0;
            }
            
            batch_idx++;
        }
    }
    
    torch::save(model, "mnist_resnet18.pt");
    std::cout << "Finished Training and Saving the model" << std::endl;
    
    return 0;
}
