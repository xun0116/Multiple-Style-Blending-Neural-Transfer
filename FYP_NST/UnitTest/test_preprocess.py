import pytest
import numpy as np
import tensorflow as tf
from PIL import Image
import io

def load_and_preprocess(img_array, max_dim=512):
    img = tf.image.convert_image_dtype(img_array, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / tf.reduce_max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)
    return tf.image.resize(img, new_shape)[tf.newaxis, :]


# --- Test Data ---
MOCK_IMAGE_ARRAY = np.random.randint(0, 256, (1200, 800, 3), dtype=np.uint8)


# --- Unit Tests ---
def test_correct_dimensions_after_resize():
    target_max_dim = 512
    preprocessed_img = load_and_preprocess(MOCK_IMAGE_ARRAY, target_max_dim)

    _, height, width, _ = preprocessed_img.shape.as_list()

    # Largest side should equal max_dim
    assert max(height, width) == target_max_dim, (
        f"Expected largest side = {target_max_dim}, got {height}x{width}"
    )

    # Aspect ratio should be preserved within 1%
    orig_h, orig_w, _ = MOCK_IMAGE_ARRAY.shape
    expected_ratio = orig_h / orig_w
    new_ratio = height / width

    assert abs(expected_ratio - new_ratio) < 0.01, (
        f"Aspect ratio mismatch: original={expected_ratio:.3f}, new={new_ratio:.3f}"
    )


def test_correct_normalization_range():
    preprocessed_img = load_and_preprocess(MOCK_IMAGE_ARRAY)

    min_val = tf.reduce_min(preprocessed_img).numpy()
    max_val = tf.reduce_max(preprocessed_img).numpy()

    # All pixels inside [0, 1]
    assert 0.0 <= min_val <= 1.0
    assert 0.0 <= max_val <= 1.0

    # Ensure dynamic range (not a blank image)
    assert min_val < 0.5, "Image too bright, lost detail"
    assert max_val > 0.5, "Image too dark, lost detail"



# !pytest -v test_preprocess.py
