// RMSNorm with the numerics of HuggingFace's Qwen2RMSNorm:
//   h   = x.float()
//   h   = h * rsqrt(mean(h*h, -1) + eps)
//   out = weight * h.to(x.dtype)        // normalized value rounded to x.dtype first
//
// CPU tensors:  hand-written single pass per row (one read of x for the sum of
//               squares, one for the output), parallel over rows.
// Other devices (CUDA): the same math composed from ATen ops inside C++. That
//               removes Python dispatch overhead per call but launches the same
//               ATen kernels as the PyTorch reference; it is NOT a custom CUDA
//               kernel.
#include <torch/extension.h>

#include <ATen/Parallel.h>

#include <cmath>
#include <type_traits>

#if defined(__AVX2__) && defined(__F16C__)
#include <immintrin.h>
#define ENGINE_HAS_F16C_PATH 1
#endif

namespace {

#ifdef ENGINE_HAS_F16C_PATH
// fp16 row, 8 lanes at a time. Same rounding sequence as the scalar path: the
// normalized value is rounded to fp16 (nearest-even), multiplied by the fp16 weight in
// fp32, and rounded to fp16 again.
void rmsnorm_row_f16c(const uint16_t* x, const uint16_t* w, uint16_t* out, int64_t hidden, float eps) {
  constexpr int kRound = _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC;
  __m256 acc = _mm256_setzero_ps();
  int64_t i = 0;
  for (; i + 8 <= hidden; i += 8) {
    const __m256 v = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i)));
    acc = _mm256_fmadd_ps(v, v, acc);
  }
  float lanes[8];
  _mm256_storeu_ps(lanes, acc);
  float sumsq = ((lanes[0] + lanes[1]) + (lanes[2] + lanes[3])) + ((lanes[4] + lanes[5]) + (lanes[6] + lanes[7]));
  for (int64_t k = i; k < hidden; ++k) {
    const float v = _cvtsh_ss(x[k]);
    sumsq += v * v;
  }
  const float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(hidden) + eps);
  const __m256 vinv = _mm256_set1_ps(inv);
  i = 0;
  for (; i + 8 <= hidden; i += 8) {
    const __m256 v = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(x + i)));
    const __m256 n = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(v, vinv), kRound));
    const __m256 wv = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w + i)));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out + i), _mm256_cvtps_ph(_mm256_mul_ps(wv, n), kRound));
  }
  for (int64_t k = i; k < hidden; ++k) {
    const float n = _cvtsh_ss(_cvtss_sh(_cvtsh_ss(x[k]) * inv, kRound));
    out[k] = _cvtss_sh(_cvtsh_ss(w[k]) * n, kRound);
  }
}
#endif

template <typename scalar_t>
void rmsnorm_rows(const scalar_t* x, const scalar_t* w, scalar_t* out,
                  int64_t rows, int64_t hidden, float eps) {
  at::parallel_for(0, rows, 1, [&](int64_t begin, int64_t end) {
    for (int64_t r = begin; r < end; ++r) {
      const scalar_t* xr = x + r * hidden;
      scalar_t* outr = out + r * hidden;

#ifdef ENGINE_HAS_F16C_PATH
      if constexpr (std::is_same_v<scalar_t, at::Half>) {
        rmsnorm_row_f16c(reinterpret_cast<const uint16_t*>(xr), reinterpret_cast<const uint16_t*>(w),
                         reinterpret_cast<uint16_t*>(outr), hidden, eps);
        continue;
      }
#endif

      // Eight independent accumulators let the compiler vectorize the reduction.
      float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
      int64_t i = 0;
      for (; i + 8 <= hidden; i += 8) {
        for (int j = 0; j < 8; ++j) {
          const float v = static_cast<float>(xr[i + j]);
          acc[j] += v * v;
        }
      }
      float sumsq = ((acc[0] + acc[1]) + (acc[2] + acc[3])) +
                    ((acc[4] + acc[5]) + (acc[6] + acc[7]));
      for (; i < hidden; ++i) {
        const float v = static_cast<float>(xr[i]);
        sumsq += v * v;
      }

      const float inv = 1.0f / std::sqrt(sumsq / static_cast<float>(hidden) + eps);
      for (int64_t k = 0; k < hidden; ++k) {
        const scalar_t normed = static_cast<scalar_t>(static_cast<float>(xr[k]) * inv);
        outr[k] = static_cast<scalar_t>(static_cast<float>(w[k]) * static_cast<float>(normed));
      }
    }
  });
}

}  // namespace

at::Tensor rmsnorm(const at::Tensor& x, const at::Tensor& weight, double eps) {
  TORCH_CHECK(x.dim() >= 1, "rmsnorm: x must have at least 1 dimension");
  TORCH_CHECK(weight.dim() == 1, "rmsnorm: weight must be 1-D");
  TORCH_CHECK(x.size(-1) == weight.size(0), "rmsnorm: last dim of x (", x.size(-1),
              ") must equal weight size (", weight.size(0), ")");
  TORCH_CHECK(x.scalar_type() == weight.scalar_type(),
              "rmsnorm: x and weight must have the same dtype");
  TORCH_CHECK(x.device() == weight.device(), "rmsnorm: x and weight must be on the same device");

  if (!x.is_cpu()) {
    at::Tensor h = x.to(at::kFloat);
    at::Tensor var = h.pow(2).mean({-1}, /*keepdim=*/true);
    h = h * at::rsqrt(var + eps);
    return weight * h.to(x.scalar_type());
  }

  at::Tensor xc = x.contiguous();
  at::Tensor wc = weight.contiguous();
  at::Tensor out = at::empty_like(xc);
  const int64_t hidden = xc.size(-1);
  const int64_t rows = hidden == 0 ? 0 : xc.numel() / hidden;
  const float eps_f = static_cast<float>(eps);

  AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, xc.scalar_type(), "rmsnorm_cpu", [&] {
    rmsnorm_rows<scalar_t>(xc.data_ptr<scalar_t>(), wc.data_ptr<scalar_t>(),
                           out.data_ptr<scalar_t>(), rows, hidden, eps_f);
  });
  return out;
}
