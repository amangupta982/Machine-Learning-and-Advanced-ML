import tensorflow as tf


class LinearRegressionModel(tf.Module):
    def __init__(self):
        super().__init__()
        self.W = tf.Variable(tf.random.normal([1]))
        self.B = tf.Variable(tf.random.normal([1]))

    def __call__(self, x):
        return self.W * x + self.B


def train_model():
    # Sample Data: y = 2x + 1
    x_train = tf.constant([[1.0], [2.0], [3.0], [4.0]])
    y_train = tf.constant([[3.0], [5.0], [7.0], [9.0]])

    model = LinearRegressionModel()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)

    for step in range(100):
        with tf.GradientTape() as tape:
            y_pred = model(x_train)
            loss = tf.reduce_mean(tf.square(y_pred - y_train))

        grads = tape.gradient(loss, [model.W, model.B])
        optimizer.apply_gradients(zip(grads, [model.W, model.B]))

    return model


if __name__ == "__main__":
    trained_model = train_model()
    print("Final Weight:", trained_model.W.numpy())
    print("Final Bias:", trained_model.B.numpy())