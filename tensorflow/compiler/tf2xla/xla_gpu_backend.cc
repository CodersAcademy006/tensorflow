/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/tf2xla_util.h"
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#include "tensorflow/core/framework/kernel_def.pb.h"

namespace tensorflow {

bool GpuOpFilter(KernelDef* kdef) {
  if (kdef->op() == "Const") {
    AddDtypeToKernelDefConstraint("dtype", DT_STRING, kdef);
  }
  if (kdef->op() == "Assert") {
    AddDtypeToKernelDefConstraint("T", DT_STRING, kdef);
  }
  // Fix for issue #105131: Ensure ConcatV2 kernels support full range of GPU types
  // instead of being incorrectly constrained to only DT_FLOAT8_E4M3FN when used
  // with Python control flow operations (filter, map, zip).
  if (kdef->op() == "ConcatV2") {
    // Check if the kernel has overly restrictive type constraints
    for (auto& constraint : *kdef->mutable_constraint()) {
      if (constraint.name() == "T") {
        auto& type_list = constraint.allowed_values().list();
        // If the constraint only has DT_FLOAT8_E4M3FN, expand it to include
        // all supported GPU types to fix the kernel constraint mismatch
        if (type_list.type_size() == 1 && 
            type_list.type(0) == DT_FLOAT8_E4M3FN) {
          // Clear the overly restrictive constraint and add all supported GPU types
          constraint.mutable_allowed_values()->mutable_list()->clear_type();
          for (const auto& dtype : kGpuAllTypes) {
            constraint.mutable_allowed_values()->mutable_list()->add_type(dtype);
          }
        }
      }
    }
  }
  return true;
}

REGISTER_XLA_BACKEND(DEVICE_GPU_XLA_JIT, kGpuAllTypes, GpuOpFilter);

}  // namespace tensorflow
