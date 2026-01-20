import tensorflow as tf 

# Input tensor
input_tensor=tf.constant([[-2.0, -1.0, 0.0, 1.0, 2.0]])

# Apply activation functions
relu_output=tf.nn.relu(input_tensor)
sigmoid_output=tf.nn.sigmoid(input_tensor)

#Output
print("Input:", input_tensor.numpy())
print("ReLU Output:", relu_output.numpy())
print("Sigmoid Output:", sigmoid_output.numpy())


"""
Observation:
- ReLU outputs zero for negative values and keeps positives unchanged.
- Sigmoid squashes values between 0 and 1.
- ReLU is preferred in hidden layers to avoid vanishing gradients.
"""