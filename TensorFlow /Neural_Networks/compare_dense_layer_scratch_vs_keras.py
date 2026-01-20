import tensorflow as tf

# ---------- Dense Layer From Scratch ----------
def dense_from_scratch(input_tensor, input_size, output_size):
    weights = tf.Variable(tf.random.normal([input_size, output_size]))
    biases = tf.Variable(tf.random.normal([output_size]))
    return tf.nn.relu(tf.matmul(input_tensor, weights) + biases)


# ---------- Dense Layer Using Keras ----------
def dense_using_keras(input_tensor, output_size):
    dense_layer = tf.keras.layers.Dense(
        output_size,
        activation="relu"
    )
    return dense_layer(input_tensor)


# ---------- Input ----------
input_tensor = tf.random.normal([1, 10])

# From scratch output
scratch_output = dense_from_scratch(input_tensor, 10, 8)

# Keras output
keras_output = dense_using_keras(input_tensor, 8)

print("From Scratch Output:", scratch_output.numpy())
print("Keras Dense Output:", keras_output.numpy())