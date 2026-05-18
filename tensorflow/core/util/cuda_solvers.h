/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_CORE_UTIL_CUDA_SOLVERS_H_
#define TENSORFLOW_CORE_UTIL_CUDA_SOLVERS_H_

#if GOOGLE_CUDA

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

class GpuEigSupport {
 public:
  explicit GpuEigSupport(OpKernelContext* context) : context_(context) {}

  // TODO: Integrate cuSOLVER geev for non-Hermitian eigendecomposition.
  Status Launch(const Tensor& input, bool compute_v, Tensor* eigenvalues,
                Tensor* eigenvectors);

 private:
  OpKernelContext* context_;
};

// Placeholder for future cuSOLVER-based eigendecomposition launch.
Status LaunchEigOnGpu(OpKernelContext* context, const Tensor& input,
                      bool compute_v, Tensor* eigenvalues,
                      Tensor* eigenvectors);

}  // namespace tensorflow

#endif  // GOOGLE_CUDA

#endif  // TENSORFLOW_CORE_UTIL_CUDA_SOLVERS_H_
