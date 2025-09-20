import pytest
import numpy as np
import tensorflow as tf

def gram_matrix(input_tensor):
    # The Gram matrix is calculated by taking the dot product of the reshaped
    # feature map with itself.
    # First, flatten the height and width dimensions.
    input_tensor = input_tensor[0]
    height, width, channels = input_tensor.shape
    num_elements = height * width
    
    reshaped_tensor = tf.reshape(input_tensor, (num_elements, channels))
    
    # Calculate the dot product of the reshaped tensor and its transpose.
    # The result will be a (channels, channels) tensor.
    gram = tf.matmul(reshaped_tensor, reshaped_tensor, transpose_a=True)
    
    # Normalize the result by dividing by the number of elements.
    return gram / tf.cast(num_elements, tf.float32)

# --- Unit Tests ---
def test_gram_matrix_calculation():
    # Create a simple tensor with a known shape and values.
    # This tensor represents a single feature map with a 2x2 area and 2 channels.
    # Shape: (1, height=2, width=2, channels=2)
    input_tensor = tf.constant([[[[1.0, 2.0], [3.0, 4.0]],
                                 [[5.0, 6.0], [7.0, 8.0]]]])
                                 
    # Manually calculate the expected Gram matrix.
    # The reshaped tensor is (4, 2):
    # [[1, 2],
    #  [3, 4],
    #  [5, 6],
    #  [7, 8]]
    
    # The dot product of the reshaped tensor and its transpose is (2, 2):
    # [[(1*1+3*3+5*5+7*7), (1*2+3*4+5*6+7*8)],
    #  [(2*1+4*3+6*5+8*7), (2*2+4*4+6*6+8*8)]]
    
    # = [[(1+9+25+49), (2+12+30+56)],
    #    [(2+12+30+56), (4+16+36+64)]]
    
    # = [[84, 100],
    #    [100, 120]]
    
    # Since the Gram matrix is normalized by the number of elements (4),
    # the expected result is:
    # [[84/4, 100/4],
    #  [100/4, 120/4]]
    
    # = [[21, 25],
    #    [25, 30]]
    expected_gram_matrix = tf.constant([[21.0, 25.0], [25.0, 30.0]])

    # Calculate the Gram matrix using the function.
    actual_gram_matrix = gram_matrix(input_tensor)

    np.testing.assert_allclose(actual_gram_matrix.numpy(), expected_gram_matrix.numpy(), rtol=1e-5)
    print("Gram matrix test passed!")

# !pytest -v test_gram_matrix.py
