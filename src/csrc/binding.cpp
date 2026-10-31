#include <torch/extension.h>

// Defined in custom_ops.cpp.
at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& weight, double eps);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Custom C++ ops for the inference engine";
  m.def("rmsnorm", &rmsnorm,
        "RMSNorm with HF Qwen2 numerics (single-pass on CPU, ATen-composed elsewhere)",
        py::arg("x"), py::arg("weight"), py::arg("eps") = 1e-6);
}
