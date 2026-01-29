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



# Output: 
#  Step 20: Loss = 19.5776
# W = [2.6100962], B = [0.83061445]
# --------------
# Step 40: Loss = 9.5584
# W = [3.4703126], B = [1.7135216]
# --------------
# Step 60: Loss = 4.2659
# W = [4.1558223], B = [2.4328842]
# --------------
# Step 80: Loss = 1.6079
# W = [4.63003], B = [2.9700346]
# --------------
# Step 100: Loss = 0.9881
# W = [4.9085546], B = [3.3403146]
# --------------

# Final Learned Values:
# W = [4.9085546]
# B = [3.3403146]