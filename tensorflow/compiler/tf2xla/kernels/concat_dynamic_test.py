# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Test for dynamic shape preservation in concat operation with XLA compilation."""

from tensorflow.python.eager import def_function
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.platform import test


class ConcatDynamicShapeTest(test.TestCase):

  def test_concat_preserves_dynamic_dimensions(self):
    """Direct test of concat with dynamic partition outputs."""

    @def_function.function(jit_compile=True)
    def test_concat_dynamic():
      x = random_ops.random_normal([8, 32])
      partitions = math_ops.cast(
          math_ops.reduce_sum(x, axis=1) > 0, dtypes.int32)

      # Create dynamic partition outputs
      partitioned = array_ops.dynamic_partition(x, partitions, num_partitions=2)

      # Concat should preserve dynamic dimension
      result = array_ops.concat(partitioned, axis=0)

      return result, array_ops.shape(result)

    result, shape = test_concat_dynamic()

    # Should succeed without errors and have reasonable output shape
    self.assertEqual(len(result.shape), 2)
    self.assertEqual(result.shape[1], 32)
    self.assertGreater(shape[0], 0)  # Dynamic first dimension


if __name__ == '__main__':
  test.main()