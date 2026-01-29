import numpy as np
import tensorflow as tf

# Create sample dataset (y = 3x + 5)
x_data = np.random.rand(200, 1).astype(np.float32)
y_data = (3 * x_data + 5).astype(np.float32)

# Trainable parameters
W = tf.Variable(tf.random.normal([1]))
B = tf.Variable(tf.random.normal([1]))

# Optimizer
optimizer = tf.keras.optimizers.Adam(learning_rate=0.05)

# Mini-batch size
batch_size = 25

# Training step in graph mode
@tf.function
def train_step(x_batch, y_batch):
    with tf.GradientTape() as tape:
        prediction = W * x_batch + B
        loss = tf.reduce_mean(tf.square(prediction - y_batch))

    gradients = tape.gradient(loss, [W, B])
    optimizer.apply_gradients(zip(gradients, [W, B]))

    return loss


# Training loop
for step in range(1, 101):

    # Select random mini-batch
    idx = np.random.choice(len(x_data), batch_size)
    x_batch = x_data[idx]
    y_batch = y_data[idx]

    # Perform one training step
    loss_value = train_step(x_batch, y_batch)

    # Print progress
    if step % 20 == 0:
        print(f"Step {step}: Loss = {loss_value.numpy():.4f}")
        print(f"W = {W.numpy()}, B = {B.numpy()}")
        print("--------------")


print("\nFinal Learned Values:")
print("W =", W.numpy())
print("B =", B.numpy())