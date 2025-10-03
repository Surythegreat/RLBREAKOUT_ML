from PIL import Image
import numpy as np

# --- Preprocessing Function ---
def preprocess_frame(frame):
    """
    Converts a frame to grayscale and resizes it to 84x84.
    """
    # Convert to grayscale using the luminosity method
    gray_frame = np.dot(frame[...,:3], [0.299, 0.587, 0.114])
    
    # Use Pillow for high-quality resizing
    pil_image = Image.fromarray(gray_frame)
    resized_image = pil_image.resize((84, 84), Image.Resampling.LANCZOS)
    
    return np.array(resized_image).astype(np.float32)

