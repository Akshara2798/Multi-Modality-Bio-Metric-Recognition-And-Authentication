# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np

# Define the main function PoC_Den that builds a DenseNet model based on input images
def PoC_Den(array_of_images):

    # Define a function for creating a dense block with multiple convolutional layers
    def dense_block(x, num_layers, growth_rate):
        for i in range(num_layers):  # Loop over the specified number of layers in the dense block
            cb = conv_block(x, growth_rate)  # Apply a convolution block for each layer
            x = layers.Concatenate()([x, cb])  # Concatenate the output of the convolution block with the input
        return x  # Return the output after the dense block

    # Define a function for a single convolutional block consisting of BN, ReLU, and Conv2D
    def conv_block(x, growth_rate):
        x = layers.BatchNormalization()(x)  # Apply BatchNormalization
        x = layers.ReLU()(x)  # Apply ReLU activation function
        x = layers.Conv2D(growth_rate, (3, 3), padding='same')(x)  # Apply a Conv2D layer with a kernel size of (3, 3)
        return x  # Return the output of the convolution block

    # Define a transition layer to reduce the number of filters and downsample the feature map
    def transition_layer(x, compression_factor):
        x = layers.BatchNormalization()(x)  # Apply BatchNormalization
        x = layers.ReLU()(x)  # Apply ReLU activation function
        num_filters = int(x.shape[-1] * compression_factor)  # Compute the number of filters after compression
        x = layers.Conv2D(num_filters, (1, 1), padding='same')(x)  # Apply a 1x1 Conv2D layer to reduce filters
        x = layers.AveragePooling2D((2, 2), strides=(2, 2))(x)  # Apply AveragePooling2D to downsample
        return x  # Return the output after the transition layer

    # Define the DenseNet model function
    def Dense_net(input_shape, num_blocks, num_layers_per_block, growth_rate, compression_factor):
        inputs = layers.Input(shape=input_shape)  # Define the input layer with the shape of input images
        x = layers.Conv2D(64, (7, 7), strides=(2, 2), padding='same')(inputs)  # Apply an initial Conv2D layer
        x = layers.BatchNormalization()(x)  # Apply BatchNormalization
        x = layers.ReLU()(x)  # Apply ReLU activation
        x = layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(x)  # Apply MaxPooling2D to downsample
        
        # Loop through the number of blocks, applying dense blocks and transition layers
        for i in range(num_blocks - 1):
            x = dense_block(x, num_layers_per_block, growth_rate)  # Apply dense block
            x = transition_layer(x, compression_factor)  # Apply transition layer
        
        # Apply the final dense block (no transition layer afterwards)
        x = dense_block(x, num_layers_per_block, growth_rate)
        x = layers.GlobalAveragePooling2D()(x)  # Apply GlobalAveragePooling2D to reduce the spatial dimensions

        model = Model(inputs, x)  # Create the Keras Model with inputs and the final output
        return model  # Return the model

    # Extract the input shape from the first image in the array
    input_shape = array_of_images[0].shape
    
    # Define parameters for the DenseNet model
    num_blocks = 4  # Number of dense blocks in the network
    num_layers_per_block = 6  # Number of convolutional layers in each block
    growth_rate = 32  # Growth rate (number of filters to add per layer)
    compression_factor = 0.5  # Compression factor for the transition layers

    # Build the DenseNet model using the defined parameters
    model = Dense_net(input_shape, num_blocks, num_layers_per_block, growth_rate, compression_factor)
    
    # Return the constructed model
    return model
