// SPDX-License-Identifier: MIT
// Original TP3 small-batch route alignment. Expert GEMM, Hadamard and T01
// clamp remain in the captured 0.8.8 implementation.
#include <ATen/core/Tensor.h>
#include <ATen/ops/empty.h>
#include <torch/csrc/utils/pybind.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

template<typename Index>
__global__ void align_decode(const Index* ids, int slots, int block_m,
    int capacity, int* sorted, int* experts, int* n_rows) {
  __shared__ int count[384];
  __shared__ int start[385];
  const int tid=threadIdx.x;
  for(int i=tid;i<capacity;i+=blockDim.x)sorted[i]=slots;
  for(int i=tid;i<capacity/block_m;i+=blockDim.x)experts[i]=-1;
  if(tid<384) {
    int c=0;
    for(int slot=0;slot<slots;slot++)if(ids[slot]==tid)c++;
    count[tid]=c;
  }
  __syncthreads();
  if(tid==0) {
    int offset=0;
    for(int e=0;e<384;e++) {start[e]=offset;offset+=((count[e]+block_m-1)/block_m)*block_m;}
    start[384]=offset;*n_rows=offset;
  }
  __syncthreads();
  if(tid<384)for(int row=start[tid];row<start[tid+1];row+=block_m)
    experts[row/block_m]=tid;
  if(tid<slots) {
    const int e=int(ids[tid]);
    if(e>=0 && e<384) {
      int local=0;
      for(int earlier=0;earlier<tid;earlier++)if(ids[earlier]==e)local++;
      sorted[start[e]+local]=tid;
    }
  }
}

std::vector<at::Tensor> align(at::Tensor ids,int64_t block_m) {
  TORCH_CHECK(ids.is_cuda() && ids.is_contiguous() && ids.dim()==2 &&
              ids.size(0)>=1 && ids.size(0)<=48 && ids.size(1)==6,"TP3 routing shape");
  TORCH_CHECK(ids.scalar_type()==at::kLong || ids.scalar_type()==at::kInt,"routing ids dtype");
  TORCH_CHECK(block_m==16 || block_m==32 || block_m==64,"routing block size");
  const int slots=int(ids.numel()),capacity=slots*int(block_m);
  auto opt=ids.options().dtype(at::kInt);
  auto sorted=at::empty({capacity},opt),expert=at::empty({capacity/int(block_m)},opt),rows=at::empty({1},opt);
  auto stream=c10::cuda::getCurrentCUDAStream();
  if(ids.scalar_type()==at::kLong)
    align_decode<int64_t><<<1,512,0,stream>>>(ids.data_ptr<int64_t>(),slots,int(block_m),capacity,
       sorted.data_ptr<int>(),expert.data_ptr<int>(),rows.data_ptr<int>());
  else align_decode<int32_t><<<1,512,0,stream>>>(ids.data_ptr<int32_t>(),slots,int(block_m),capacity,
       sorted.data_ptr<int>(),expert.data_ptr<int>(),rows.data_ptr<int>());
  auto error=cudaGetLastError();
  TORCH_CHECK(error==cudaSuccess,"routing launch failed: ",cudaGetErrorString(error));
  return {sorted,expert,rows};
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("align",&align);}
