import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Generate data
x_data = np.random.normal(1, 0.1, 100)
y_data = np.repeat(10., 100)

# Trainable parameter
C = tf.Variable(tf.random.normal(shape=[1, 1], dtype='float64'))

# Optimizer
bt_opt = tf.keras.optimizers.SGD(learning_rate=0.02)

# Batch settings
batch_size = 20
loss_batch = []

# Training loop
for i in range(100):

    # Select random batch
    rand_index = np.random.choice(100, size=batch_size)
    rand_x = np.transpose([x_data[rand_index]])
    rand_y = np.transpose([y_data[rand_index]])

    # Forward + Loss
    with tf.GradientTape() as tape:
        output = tf.matmul(rand_x, C)
        cur_loss = tf.reduce_mean(tf.square(output - rand_y))

    # Backpropagation
    grads = tape.gradient(cur_loss, [C])
    bt_opt.apply_gradients(zip(grads, [C]))

    # Store loss every 5 steps
    if (i + 1) % 5 == 0:
        loss_batch.append(cur_loss.numpy())

    # Print progress every 25 steps
    if (i + 1) % 25 == 0:
        print("Step:", i + 1)
        print("C =", C.numpy())
        print("Loss =", cur_loss.numpy())
        print("---------------")

# Plot batch loss
plt.plot(range(0, 100, 5), loss_batch, label="Batch Loss")
plt.legend()
plt.show()


# OUTPUT: 
# Step: 25
# C = [[6.41558351]]
# Loss = 15.196331333729933
# ---------------
# Step: 50
# C = [[8.70551768]]
# Loss = 2.342073307467174
# ---------------
# Step: 75
# C = [[9.5545582]]
# Loss = 1.5266569412507704
# ---------------
# Step: 100
# C = [[9.85123305]]
# Loss = 1.2559391600585568
# ---------------