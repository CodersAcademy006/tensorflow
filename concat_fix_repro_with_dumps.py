import tensorflow as tf
import os

# Enable XLA dumps
os.environ['TF_DUMP_GRAPH_PREFIX'] = '/workspaces/tensorflow/xla_dump'
os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_lazy_compilation=false'

class TestModel(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.d1 = tf.keras.layers.Dense(64, activation='relu')
        self.d2 = tf.keras.layers.Dense(32)
        self.d3 = tf.keras.layers.Dense(16)

    def call(self, x, indices=None):
        x = self.d1(x)
        if indices is not None:
            (unique_vals, _) = tf.unique(indices)
            x = tf.nn.relu(tf.gather(x, unique_vals))
        else:
            x = tf.nn.relu(x)
        partitioned = tf.dynamic_partition(x, tf.cast(tf.reduce_sum(x, axis=1) > 0, tf.int32), num_partitions=2)
        x = tf.concat(partitioned, axis=0)
        (top_k_values, _) = tf.nn.top_k(x, k=tf.shape(x)[0] // 2)
        x = tf.nn.relu(self.d2(top_k_values))
        return self.d3(x)

def main():
    model = TestModel()
    x = tf.random.normal([10, 64])
    indices = tf.random.uniform([10], maxval=5, dtype=tf.int32)
    
    print('Running eager mode...')
    eager_out = model(x, indices)
    print('Eager Output shape:', eager_out.shape)
    
    print('Running XLA compilation...')
    @tf.function(jit_compile=True)
    def compiled_forward(x_input, indices_input):
        return model(x_input, indices_input)
    
    try:
        compiled_out = compiled_forward(x, indices)
        print('XLA Output shape:', compiled_out.shape)
    except Exception as e:
        print('XLA compilation failed:')
        print(str(e))
        return

if __name__ == '__main__':
    main()