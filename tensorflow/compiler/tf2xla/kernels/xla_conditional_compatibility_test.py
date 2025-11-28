#!/usr/bin/env python3
# Copyright 2025 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for XLA conditional compatibility with Python control flow operations.

This test addresses GitHub issue #105131: ConcatV2 op kernel constraint
mismatch when using jit_compile=True with Python control flow (filter,
map, zip).
"""

from absl.testing import parameterized

import tensorflow as tf
from tensorflow.python.platform import test
from tensorflow.python.framework import dtypes


class XlaConditionalCompatibilityTest(test.TestCase, parameterized.TestCase):
  """Test XLA compilation with Python control flow and concat operations."""

  def setUp(self):
    super().setUp()
    # Enable XLA JIT compilation for tests
    tf.config.optimizer.set_jit(True)

  @parameterized.parameters([
      (dtypes.float32,),
      (dtypes.float16,), 
      (dtypes.bfloat16,),
      (dtypes.int32,),
      (dtypes.int64,),
  ])
  def test_python_filter_with_concat_xla(self, dtype):
    """Test Python filter operations with tf.concat under XLA compilation."""
    
    @tf.function(jit_compile=True)
    def filter_and_concat(x):
      # Use filter to get tensors above threshold
      filtered = list(filter(
          lambda z: tf.reduce_sum(z) > 0.5, [x, x * 2, x * 3]))
      if not filtered:
        return tf.zeros_like(x)
      return tf.concat(filtered, axis=-1)
    
    # Test with different input shapes and dtypes
    if dtype.is_integer:
      input_tensor = tf.constant([[1, 2], [3, 4]], dtype=dtype)
    else:
      input_tensor = tf.constant([[0.1, 0.2], [0.3, 0.4]], dtype=dtype)
    
    # This should not raise a ConcatV2 kernel constraint error
    result = filter_and_concat(input_tensor)
    self.assertIsNotNone(result)
    
  @parameterized.parameters([
      (dtypes.float32,),
      (dtypes.float16,),
      (dtypes.bfloat16,),
      (dtypes.int32,),
  ])
  def test_python_map_with_concat_xla(self, dtype):
    """Test Python map operations with tf.concat under XLA compilation."""
    
    @tf.function(jit_compile=True)
    def map_and_concat(x):
      # Use Python map with tf.concat
      features = [x, x + 1, x * 2]
      mapped_features = list(map(
          lambda z: tf.nn.sigmoid(z) if not dtype.is_integer else z,
          features))
      return tf.concat(mapped_features, axis=-1)
    
    if dtype.is_integer:
      input_tensor = tf.constant([[1, 2], [3, 4]], dtype=dtype)
    else:
      input_tensor = tf.constant([[0.1, 0.2], [0.3, 0.4]], dtype=dtype)
    
    # This should not raise a ConcatV2 kernel constraint error
    result = map_and_concat(input_tensor)
    self.assertIsNotNone(result)
    
  @parameterized.parameters([
      (dtypes.float32,),
      (dtypes.int32,),
  ])
  def test_python_zip_with_concat_xla(self, dtype):
    """Test Python zip operations with tf.concat under XLA compilation."""
    
    @tf.function(jit_compile=True)
    def zip_and_concat(x):
      # Use Python zip with tf.concat - this was part of the original issue
      features = [x, x * 2, x + 1]
      constants = [tf.ones_like(x) for _ in range(len(features))]
      zipped_data = list(zip(features, constants))
      
      # Flatten the zipped pairs and concatenate
      flattened = []
      for feature, constant in zipped_data:
        flattened.extend([feature, constant])
      
      return tf.concat(flattened, axis=-1)
    
    if dtype.is_integer:
      input_tensor = tf.constant([[1, 2], [3, 4]], dtype=dtype)
    else:
      input_tensor = tf.constant([[0.1, 0.2], [0.3, 0.4]], dtype=dtype)
    
    # This should not raise a ConcatV2 kernel constraint error  
    result = zip_and_concat(input_tensor)
    self.assertIsNotNone(result)

  def test_github_issue_105131_reproduction(self):
    """Test the exact scenario from GitHub issue #105131."""
    
    class TestModel(tf.keras.Model):
      def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(32, activation='relu')
        self.d2 = tf.keras.layers.Dense(16, activation='tanh')
        self.d3 = tf.keras.layers.Dense(8)

      def call(self, x):
        # This exact pattern from the GitHub issue
        filtered_features = list(filter(
            lambda z: tf.reduce_sum(z) > 0.5, [x, x * 2, x * 3]))
        mapped_features = list(map(
            lambda z: tf.nn.sigmoid(z), filtered_features))
        zipped_data = list(zip(
            mapped_features,
            [tf.ones_like(x) for _ in range(len(mapped_features))]))
        # Flatten the zipped tuples for concat
        flattened = []
        for pair in zipped_data:
          flattened.extend(pair)
        combined = tf.concat(flattened, axis=-1)
        return self.d3(combined)

    model = TestModel()
    x = tf.random.normal([4, 16])
    
    # Test eager execution first (should work)
    eager_out = model(x)
    self.assertIsNotNone(eager_out)
    
    # Test XLA compilation (this was previously failing)
    @tf.function(jit_compile=True)
    def compiled_forward(inputs):
      return model(inputs)
    
    # Check this doesn't raise OpKernel constraint error
    xla_out = compiled_forward(x)
    self.assertIsNotNone(xla_out)
    
    # Results should have the same shape
    self.assertEqual(eager_out.shape, xla_out.shape)

  def test_concat_with_mixed_control_flow_patterns(self):
    """Test complex combinations of control flow with concat operations."""
    
    @tf.function(jit_compile=True)
    def complex_control_flow_concat(x):
      # Combine filter, map, and zip in a single function
      # This tests the most complex scenario that could trigger the bug
      
      # Step 1: Filter
      base_features = [x, x * 2, x * 3, x + 1]
      filtered = list(filter(
          lambda z: tf.reduce_sum(tf.abs(z)) > 0.1, base_features))
      
      # Step 2: Map
      mapped = list(map(lambda z: tf.nn.tanh(z), filtered))
      
      # Step 3: Zip with constants
      constants = [tf.ones_like(x) * i for i in range(len(mapped))]
      zipped = list(zip(mapped, constants))
      
      # Step 4: Flatten and concat (the critical operation)
      final_features = []
      for feature, constant in zipped:
        final_features.append(feature)
        final_features.append(constant)
      
      return tf.concat(final_features, axis=-1)
    
    x = tf.random.normal([2, 4])
    
    # This comprehensive test should pass with the fix
    result = complex_control_flow_concat(x)
    self.assertIsNotNone(result)
    
    # Verify the result has expected properties
    self.assertEqual(len(result.shape), 2)
    self.assertEqual(result.shape[0], 2)  # batch dimension preserved


if __name__ == '__main__':
  test.main()