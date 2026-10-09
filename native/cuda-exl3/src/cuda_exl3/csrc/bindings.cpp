#include <torch/extension.h>
#include <torch/library.h>

namespace cuda_exl3 {
void exl3_had_in(const at::Tensor& x, at::Tensor& out, const at::Tensor& suh);
void exl3_reserve(const at::Tensor& like, int64_t max_tokens, int64_t k, int64_t n,
                  int64_t groups);
void exl3_moe_had_in(const at::Tensor& x, at::Tensor& out, const at::Tensor& suh,
                     const at::Tensor& sorted_ids, const at::Tensor& expert_ids,
                     const at::Tensor& n_rows, int64_t block_m, int64_t top_k,
                     int64_t m_valid, bool skip_padding);
void exl3_reserve_acc(const at::Tensor& like, int64_t elems);
void exl3_set_moe_acc_cap(int64_t elems);
int64_t exl3_get_moe_acc_cap();
void exl3_moe_glu_had_in(const at::Tensor& x, at::Tensor& out, const at::Tensor& suh,
                         const at::Tensor& expert_ids, const at::Tensor& n_rows,
                         int64_t block_m);
at::Tensor exl3_moe_combine(const at::Tensor& rows_out, const at::Tensor& sorted_ids,
                            const at::Tensor& topk_weights, int64_t num_tokens,
                            const std::optional<at::Tensor>& expert_ids,
                            int64_t block_m);
at::Tensor exl3_moe_gemm(const at::Tensor& a_had, const at::Tensor& trellis,
                         const at::Tensor& suh_unused, const at::Tensor& svh,
                         const at::Tensor& expert_ids, const at::Tensor& n_rows,
                         at::IntArrayRef group_n, int64_t cb, int64_t block_m,
                         at::ScalarType out_dtype,
                         const std::optional<at::Tensor>& sorted_ids,
                         const std::optional<at::Tensor>& topk_weights,
                         int64_t num_tokens, int64_t top_k);
at::Tensor mla_decode(const at::Tensor& q, const at::Tensor& kv,
                      const at::Tensor& sel, const at::Tensor& seqlens,
                      double scale, int64_t v_head_dim, int64_t split_chunk,
                      int64_t heads_per_block, double kv_scale);
at::Tensor exl3_linear(const at::Tensor& x, const at::Tensor& trellis,
                       const at::Tensor& suh, const at::Tensor& svh,
                       at::IntArrayRef group_n, int64_t cb, bool split_k);
}  // namespace cuda_exl3

// Registered through torch.library rather than raw pybind so that torch.compile
// / Dynamo can trace the call (a bare pybind function is opaque to it, which
// blocks vLLM's CUDA graph capture). The op is functional -- it allocates and
// returns its output -- which also keeps it easy to give a meta implementation.
TORCH_LIBRARY(cuda_exl3_C, m)
{
    m.def(
        "exl3_linear(Tensor x, Tensor trellis, Tensor suh, Tensor svh, "
        "int[] group_n, int cb, bool split_k) -> Tensor");
    m.def("exl3_had_in(Tensor x, Tensor(a!) out, Tensor suh) -> ()");
    m.def("exl3_reserve(Tensor like, int max_tokens, int k, int n, int groups) -> ()");
    m.def("exl3_reserve_acc(Tensor like, int elems) -> ()");
    m.def("exl3_set_moe_acc_cap(int elems) -> ()");
    m.def("exl3_get_moe_acc_cap() -> int");
    m.def(
        "exl3_moe_had_in(Tensor x, Tensor(a!) out, Tensor suh, Tensor sorted_ids, "
        "Tensor expert_ids, Tensor n_rows, int block_m, int top_k, int m_valid, "
        "bool skip_padding=False) -> ()");
    m.def(
        "exl3_moe_glu_had_in(Tensor x, Tensor(a!) out, Tensor suh, Tensor expert_ids, "
        "Tensor n_rows, int block_m) -> ()");
    m.def(
        "exl3_moe_gemm(Tensor a_had, Tensor trellis, Tensor suh, Tensor svh, "
        "Tensor expert_ids, Tensor n_rows, int[] group_n, int cb, int block_m, "
        "ScalarType out_dtype, Tensor? sorted_ids=None, Tensor? topk_weights=None, "
        "int num_tokens=0, int top_k=0) -> Tensor");
    m.def(
        "exl3_moe_combine(Tensor rows_out, Tensor sorted_ids, Tensor topk_weights, "
        "int num_tokens, Tensor? expert_ids=None, int block_m=0) -> Tensor");
    m.def(
        "mla_decode(Tensor q, Tensor kv, Tensor sel, Tensor seqlens, float scale, "
        "int v_head_dim, int split_chunk, int heads_per_block, float kv_scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(cuda_exl3_C, CUDA, m)
{
    m.impl("exl3_linear", &cuda_exl3::exl3_linear);
    m.impl("exl3_had_in", &cuda_exl3::exl3_had_in);
    m.impl("exl3_reserve", &cuda_exl3::exl3_reserve);
    m.impl("exl3_reserve_acc", &cuda_exl3::exl3_reserve_acc);
    m.impl("exl3_moe_had_in", &cuda_exl3::exl3_moe_had_in);
    m.impl("exl3_moe_glu_had_in", &cuda_exl3::exl3_moe_glu_had_in);
    m.impl("exl3_moe_gemm", &cuda_exl3::exl3_moe_gemm);
    m.impl("exl3_moe_combine", &cuda_exl3::exl3_moe_combine);
    m.impl("mla_decode", &cuda_exl3::mla_decode);
}

TORCH_LIBRARY_IMPL(cuda_exl3_C, CompositeExplicitAutograd, m)
{
    m.impl("exl3_set_moe_acc_cap", &cuda_exl3::exl3_set_moe_acc_cap);
    m.impl("exl3_get_moe_acc_cap", &cuda_exl3::exl3_get_moe_acc_cap);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.doc() = "EXL3 kernels; ops are registered under torch.ops.cuda_exl3_C";
}
