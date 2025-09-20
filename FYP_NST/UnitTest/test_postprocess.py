import pytest
import numpy as np
import tensorflow as tf

def post_process_image(img_tensor, brightness=0.0, color_preserve_target=None):
    # Clamp values to the [0, 1] range after processing
    # and add brightness adjustment
    processed_img = tf.clip_by_value(img_tensor + brightness, 0.0, 1.0)
    
    if color_preserve_target is not None:
        yuv_styled = tf.image.rgb_to_yuv(processed_img)
        yuv_target = tf.image.rgb_to_yuv(color_preserve_target)
        
        # Combine luminance from styled image with color from target image
        yuv_output = tf.concat([yuv_styled[..., 0:1], yuv_target[..., 1:3]], axis=-1)
        
        processed_img = tf.image.yuv_to_rgb(yuv_output)
        processed_img = tf.clip_by_value(processed_img, 0.0, 1.0)
        
    return processed_img

# --- Unit Tests ---

def test_brightness_adjustment():
    # Create a mock image tensor with known values
    original_image = tf.constant([[[[0.5, 0.5, 0.5]]]], dtype=tf.float32)
    
    # A positive brightness adjustment
    brightness_value = 0.2
    
    # Apply the post-processing
    processed_image = post_process_image(original_image, brightness=brightness_value)
    
    # The expected new value is 0.5 + 0.2 = 0.7
    expected_value = 0.7
    

    np.testing.assert_allclose(processed_image.numpy(), expected_value, rtol=1e-5)
    
def test_color_preservation():

    # Create a mock styled image (e.g., has a green tint)
    styled_image = tf.constant([[[[0.2, 0.8, 0.2]]]], dtype=tf.float32)
    
    # Create a mock content image with a strong red tint, to test color transfer
    target_image = tf.constant([[[[0.8, 0.2, 0.2]]]], dtype=tf.float32)
    
    # Apply the post-processing with color preservation
    processed_image = post_process_image(styled_image, color_preserve_target=target_image)
    
    # Calculate the average color 
    # The output image's color should be closer to the target (red) than the styled (green).
    
    # The red channel of the output image should be higher than the red channel
    # of the styled image, and its green channel should be lower.
    
    # Check the red channel
    styled_red = styled_image[..., 0].numpy()
    processed_red = processed_image[..., 0].numpy()
    
    # Check the green channel
    styled_green = styled_image[..., 1].numpy()
    processed_green = processed_image[..., 1].numpy()
    
    # The processed image should have a higher red value and a lower green value
    # compared to the original styled image.
    assert processed_red > styled_red
    assert processed_green < styled_green

# !pytest -v test_postprocess.py