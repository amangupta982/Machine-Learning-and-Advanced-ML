# Important library to import

import tensorflow as tf 
from tensorflow.keras import models,layers 

model= models.Sequential([
    layers.Dense(8,activation='relu',input_shape=(10,)), #HiddenLayer 1
    layers.Dense(5,activation='relu'),   #HiddenLayer 2
    layers.Dense(2,activation='relu')   #Output Layer 
])

# Create a random input (batch size = 1, 10 features)
input_tensor=tf.random.normal([1,10])

# Forward pass
output=model(input_tensor)

# Print outputs
print("Model Output:",output.numpy())