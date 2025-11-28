#!/usr/bin/env python3
"""
Reproduction script for GitHub issue #105131:
[XLA/GPU] ConcatV2 op kernel constraint mismatch when using jit_compile=True with Python control flow

This script reproduces the ConcatV2 kernel constraint error that occurs when:
1. Using @tf.function(jit_compile=True) 
2. Python control flow operations (filter, map, zip)
3. Followed by tf.concat
"""

import tensorflow as tf
import sys
class TestModel(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(32, activation='relu')
        self.d2 = tf.keras.layers.Dense(16, activation='tanh')
        self.d3 = tf.keras.layers.Dense(8)

    def call(self, x):
        # This combination of Python control flow + tf.concat causes the issue
        filtered_features = list(filter(lambda z: tf.reduce_sum(z) > 0.5, [x, x * 2, x * 3]))
        mapped_features = list(map(lambda z: tf.nn.sigmoid(z), filtered_features))
        zipped_data = list(zip(mapped_features, [tf.ones_like(x) for _ in range(len(mapped_features))]))
        
        print(f"Debug: zipped_data length: {len(zipped_data)}")
        if zipped_data:
            print(f"Debug: zipped_data[0] types: {[type(item) for item in zipped_data[0]]}")
            print(f"Debug: zipped_data[0] dtypes: {[item.dtype if hasattr(item, 'dtype') else 'N/A' for item in zipped_data[0]]}")
        
        # This tf.concat call triggers the ConcatV2 kernel constraint error
        combined = tf.concat(zipped_data, axis=-1)
        return self.d3(combined)

def get_default_model():
    return TestModel()

def get_sample_inputs():
    x = tf.random.normal([4, 16])
    return (x,)

def main():
    print("=== XLA ConcatV2 Kernel Constraint Issue Reproduction ===")
    print(f"TensorFlow version: {tf.__version__}")
    
    model = get_default_model()
    inputs = get_sample_inputs()
    
    # Test eager execution first (should work)
    print("\n1. Testing eager execution...")
    try:
        eager_out = model(*inputs)
        print(f'✓ Eager Input shape: {inputs[0].shape}')
        print(f'✓ Eager Output shape: {eager_out.shape}')
    except Exception as e:
        print(f'✗ Eager execution failed: {e}')
        return 1
    
    # Test XLA compilation (should fail with kernel constraint error)
    print("\n2. Testing XLA compilation...")
    @tf.function(jit_compile=True)
    def compiled_forward(*args):
        return model(*args)
    
    try:
        compiled_out = compiled_forward(*inputs)
        print(f'✓ XLA Output shape: {compiled_out.shape}')
        print("✓ Issue appears to be resolved!")
        return 0
    except Exception as e:
        print(f'✗ XLA compilation failed with error: {e}')
        print("\nThis confirms the ConcatV2 kernel constraint issue exists.")
        return 1

if __name__ == '__main__':
    sys.exit(main())