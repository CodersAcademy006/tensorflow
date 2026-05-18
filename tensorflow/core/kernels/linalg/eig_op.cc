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

#include "absl/base/call_once.h"
#include "absl/strings/ascii.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/device_name_utils.h"

namespace tensorflow {
namespace {

bool RequestedGpuPlacement(const OpKernel& kernel) {
  const std::string& requested_device = kernel.requested_device();
  if (requested_device.empty()) {
    return false;
  }
  DeviceNameUtils::ParsedName parsed;
  if (!DeviceNameUtils::ParseFullOrLocalName(requested_device, &parsed)) {
    return false;
  }
  return parsed.has_type &&
         absl::EqualsIgnoreCase(parsed.type, DEVICE_GPU);
}

}  // namespace

void MaybeWarnOnGpuFallback(const OpKernel& kernel) {
  if (!RequestedGpuPlacement(kernel)) {
    return;
  }
  if (KernelDefAvailable(DeviceType(DEVICE_GPU), kernel.def())) {
    return;
  }
  static absl::once_flag warn_once;
  absl::call_once(warn_once, [] {
    LOG(WARNING)
        << "tf.linalg.eig has no registered GPU kernel and will execute on CPU."
        << " This may incur host-device transfer overhead.";
  });
}

}  // namespace tensorflow
