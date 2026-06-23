#include <ATen/ATen.h>
#include <c10/core/DeviceType.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

#include <iostream>
#include <string>
#include <vector>

namespace {

constexpr int kFeatureCount = 311;

} // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " MODEL_PACKAGE.pt2\n";
    return 2;
  }

  const std::string package_path = argv[1];
  torch::inductor::AOTIModelPackageLoader loader(
      package_path,
      "model",
      false,
      1,
      -1);

  auto input = at::zeros(
      {1, kFeatureCount},
      at::TensorOptions().dtype(at::kFloat).device(at::kMPS));
  std::vector<at::Tensor> outputs = loader.run({input});

  std::cout << "outputs=" << outputs.size() << "\n";
  for (std::size_t index = 0; index < outputs.size(); ++index) {
    const at::Tensor& output = outputs[index];
    std::cout << "output[" << index << "] sizes=" << output.sizes()
              << " device=" << output.device() << "\n";
  }
  std::cout << "policy_first=" << outputs[0][0][0].item<float>() << "\n";
  std::cout << "value=" << outputs[1][0].item<float>() << "\n";
  return 0;
}
