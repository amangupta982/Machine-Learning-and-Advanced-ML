import tensorflow as tf

# ---------- Dense Layer ----------
class DenseFromScratch:
    def __init__(self, input_size, output_size):
        self.weights = tf.Variable(
            tf.random.normal([input_size, output_size])
        )
        self.biases = tf.Variable(
            tf.zeros([output_size])
        )

    def forward(self, x):
        return tf.matmul(x, self.weights) + self.biases


# ---------- Mean Squared Error ----------
def mse_loss(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


# ---------- Training Step ----------
def train_step(model, x, y, learning_rate=0.01):
    with tf.GradientTape() as tape:
        predictions = model.forward(x)
        loss = mse_loss(y, predictions)

    gradients = tape.gradient(
        loss, [model.weights, model.biases]
    )

    model.weights.assign_sub(learning_rate * gradients[0])
    model.biases.assign_sub(learning_rate * gradients[1])

    return loss


# ---------- Dummy Data ----------
x = tf.random.normal([5, 3])   # 5 samples, 3 features
y = tf.random.normal([5, 1])   # target

# ---------- Model ----------
model = DenseFromScratch(input_size=3, output_size=1)

# ---------- Training Loop ----------
for epoch in range(10):
    loss = train_step(model, x, y)
    print(f"Epoch {epoch+1}, Loss: {loss.numpy():.4f}")