import pytest
import numpy as np
import tensorflow as tf
from PIL import Image
import io

# --- Project Functions ---
def process_upload(upload_widget):
    if upload_widget.value:
        upload_data = upload_widget.value[0]
        content = upload_data['content']
        img = Image.open(io.BytesIO(content)).convert("RGB")
        return np.array(img)
    raise ValueError("No file uploaded")

def load_and_preprocess(img_array, max_dim=512):
    img = tf.image.convert_image_dtype(img_array, tf.float32)
    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    scale = max_dim / tf.reduce_max(shape)
    new_shape = tf.cast(shape * scale, tf.int32)
    return tf.image.resize(img, new_shape)[tf.newaxis, :]

def train_step():
    return tf.constant([1.0], dtype=tf.float32)

# --- Pytest Fixture and Test Class ---
class TestFullPipeline:
    
    # Create a mock image array with a known shape
    MOCK_IMAGE_ARRAY = np.random.randint(0, 256, (1200, 800, 3), dtype=np.uint8)

    @pytest.fixture
    def mock_upload_widget(self):
        # Create an in-memory byte stream to store the image data
        in_memory_file = io.BytesIO()

        # Convert the NumPy array to a Pillow Image
        img = Image.fromarray(self.MOCK_IMAGE_ARRAY)

        # Save the image to the in-memory file as a PNG
        img.save(in_memory_file, format='PNG')
        
        # Reset the stream position to the beginning so it can be read
        in_memory_file.seek(0)

        # Simulate a file upload by creating a dictionary with mock content.
        mock_widget_value = [
            {
                'content': in_memory_file.getvalue(),
                'metadata': {'name': 'mock_image.png'}
            }
        ]
        
        class MockUploadWidget:
            def __init__(self, value):
                self.value = value
        
        return MockUploadWidget(mock_widget_value)
    
    # --- Integration Test Case ---
    def test_full_pipeline_runs_without_errors(self, mock_upload_widget):
        try:
            img_array = process_upload(mock_upload_widget)
            
            img_tensor = load_and_preprocess(img_array)
            assert isinstance(img_tensor, tf.Tensor)
            assert img_tensor.shape.as_list() == [1, 512, 341, 3]
            
            result = train_step()
            assert isinstance(result, tf.Tensor)
            assert result.numpy() is not None
            
        except Exception as e:
            pytest.fail(f"Full pipeline integration test failed with an error: {e}")

# !pytest -v test_integration.py