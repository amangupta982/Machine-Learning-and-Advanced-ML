import tensorflow as tf
from linear_regression_model import train_model


class LinearRegressionTest(tf.test.TestCase):

    def test_model_training(self):
        model = train_model()

        # Check learned values are close to expected
        self.assertAllClose(model.W.numpy(), [2.0], atol=0.2)
        self.assertAllClose(model.B.numpy(), [1.0], atol=0.2)

    def test_prediction_output(self):
        model = train_model()

        x_test = tf.constant([[5.0]])
        prediction = model(x_test)

        # Expected: y = 2*5 + 1 = 11
        self.assertAllClose(prediction.numpy(), [[11.0]], atol=0.5)


if __name__ == "__main__":
    tf.test.main()