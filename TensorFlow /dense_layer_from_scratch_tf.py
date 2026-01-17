import tensorflow as tf

# Define a function to create a neural network layer
def neural_network(input_tensor,input_size,output_size):
    
    # Step 1: Initialize weights
    weights= tf.Variable(tf.random.normal([input_size,output_size]))
    
    # Step 2: Initialize biases
    biases=tf.Variable(tf.random.normal([output_size]))
    
    # Step 3: Linear transformation + ReLU activation
    return tf.nn.relu(tf.matmul(input_tensor,weights)+ biases)


# Input layer: batch size = 1, input size = 10 features
input_tensor= tf.random.normal([1,10])

# Hidden layer 1: 10 → 8 neurons
hidden_layer_1=neural_network(input_tensor,10,8)

# Hidden layer 2: 8 → 5 neurons 
hidden_layer_2=neural_network(hidden_layer_1,8,5)

# Output layer: 5 → 2 neurons
output_layer=neural_network(hidden_layer_2,5,2)

# Print outputs
print("Hidden Layer 1 Output:", hidden_layer_1.numpy())
print("Hidden Layer 2 Output:", hidden_layer_2.numpy())
print("Output Layer:", output_layer.numpy())