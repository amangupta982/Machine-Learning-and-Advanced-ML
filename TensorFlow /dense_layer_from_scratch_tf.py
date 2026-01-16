import tensorflow as tf 

# Define a function to create a neural network layer
def neural_network(input_tensor,input_size,output_size):
    
    # Step 1: Initialize weights
    weights= tf.Variable(tf.random.normal([input_size,output_size]))
    
    # Step 2: Initialize biases
    biases=tf.Variable(tf.random.normal([output_size]))
    
    # Step 3: Linear transformation + ReLU activation
    return tf.nn.relu(tf.matmul(input_tensor,weights)+ biases)
