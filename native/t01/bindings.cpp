#include <torch/extension.h>
#include <torch/library.h>
namespace cuda_exl3 {
void exl3_moe_glu_had_in_t01(const at::Tensor&,at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,int64_t,double);
}
TORCH_LIBRARY_FRAGMENT(cuda_exl3_C,m) {
 m.def("exl3_moe_glu_had_in_t01(Tensor x, Tensor(a!) out, Tensor suh, Tensor expert_ids, Tensor n_rows, int block_m, float limit=0.) -> ()");
}
TORCH_LIBRARY_IMPL(cuda_exl3_C,CUDA,m) {m.impl("exl3_moe_glu_had_in_t01",&cuda_exl3::exl3_moe_glu_had_in_t01);}
