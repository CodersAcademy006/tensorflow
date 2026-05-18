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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#include "tensorflow/core/framework/kernel_def_builder.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/linalg/linalg_ops_common.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace {

template <typename Scalar>
class EigOpGpu : public OpKernel {
 public:
  explicit EigOpGpu(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // TODO: Wire this to cuSOLVER geev via GpuEigSupport/LaunchEigOnGpu.
    context->SetStatus(errors::Unimplemented(
        "GPU kernel for tf.linalg.eig is not yet implemented; CPU fallback "
        "disabled for explicit visibility."));
  }
};

}  // namespace

#define REGISTER_GPU(Scalar) \
  REGISTER_LINALG_OP_GPU("Eig", (EigOpGpu<Scalar>), Scalar)

REGISTER_GPU(float);
REGISTER_GPU(double);
REGISTER_GPU(complex64);
REGISTER_GPU(complex128);

#undef REGISTER_GPU

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
