import tensorflow as tf

# A simple regular Python function
def a_regular_function(x, y, b):
    return tf.matmul(x, y) + b

# Convert it into a TensorFlow Function (Graph mode)
a_function_that_uses_a_graph = tf.function(a_regular_function)

# Make some tensors
x1 = tf.constant([[1.0, 2.0]])
y1 = tf.constant([[2.0], [3.0]])
b1 = tf.constant(4.0)

# Normal Python function call
orig_value = a_regular_function(x1, y1, b1).numpy()

# TensorFlow Function call
tf_function_value = a_function_that_uses_a_graph(x1, y1, b1).numpy()

print("Original Function Output:", orig_value)
print("TensorFlow Function Output:", tf_function_value)

#OUTPUT:
# Original Function Output: [[12.]]
# TensorFlow Function Output: [[12.]]