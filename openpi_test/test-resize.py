import numpy as np
from typing import Union, Literal
import warnings

def resize_with_pad(
    images: Union[np.ndarray, np.ndarray],
    height: int,
    width: int,
    method: Literal['linear', 'nearest', 'cubic'] = 'linear'
) -> np.ndarray:
    """Replicates tf.image.resize_with_pad. Resizes an image to a target height and width without distortion
    by padding with black. If the image is float32, it must be in the range [-1, 1].
    
    Args:
        images: Input image(s) as numpy array with shape (height, width, channels) or 
                (batch, height, width, channels). Can be uint8 or float32.
        height: Target height.
        width: Target width.
        method: Resize method: 'linear', 'nearest', or 'cubic'.
    
    Returns:
        Resized and padded image(s) as numpy array.
    """
    # Store original shape and dtype
    original_shape = images.shape
    original_dtype = images.dtype
    has_batch_dim = images.ndim == 4
    
    # Ensure we have a batch dimension for consistent processing
    if not has_batch_dim:
        images = images[None, ...]
    
    batch_size, cur_height, cur_width, channels = images.shape
    
    # Calculate resize ratio to fit within target dimensions
    width_ratio = cur_width / width
    height_ratio = cur_height / height
    ratio = max(width_ratio, height_ratio)
    
    # Calculate new dimensions after resizing
    resized_height = int(np.round(cur_height / ratio))
    resized_width = int(np.round(cur_width / ratio))
    
    # Ensure minimum dimensions of 1
    resized_height = max(1, resized_height)
    resized_width = max(1, resized_width)
    
    # Resize each image in the batch
    resized_images = []
    for i in range(batch_size):
        img = images[i]
        
        if method == 'nearest':
            # Nearest neighbor interpolation
            x_ratio = cur_width / resized_width
            y_ratio = cur_height / resized_height
            x_idx = (np.arange(resized_width) * x_ratio).astype(np.int32)
            y_idx = (np.arange(resized_height) * y_ratio).astype(np.int32)
            x_idx = np.clip(x_idx, 0, cur_width - 1)
            y_idx = np.clip(y_idx, 0, cur_height - 1)
            
            # Use advanced indexing
            resized_img = img[np.ix_(y_idx, x_idx, range(channels))]
            
        elif method == 'linear':
            # Bilinear interpolation
            # Create grid of target coordinates
            x = np.linspace(0, cur_width - 1, resized_width)
            y = np.linspace(0, cur_height - 1, resized_height)
            
            # Get integer coordinates and weights
            x0 = np.floor(x).astype(np.int32)
            x1 = np.minimum(x0 + 1, cur_width - 1)
            y0 = np.floor(y).astype(np.int32)
            y1 = np.minimum(y0 + 1, cur_height - 1)
            
            # Calculate interpolation weights
            wx = x - x0
            wy = y - y0
            wx = wx.reshape(1, -1, 1)
            wy = wy.reshape(-1, 1, 1)
            
            # Perform bilinear interpolation for each channel
            resized_img = np.zeros((resized_height, resized_width, channels), dtype=images.dtype)
            
            for c in range(channels):
                I00 = img[y0[:, None], x0[None, :], c]
                I01 = img[y0[:, None], x1[None, :], c]
                I10 = img[y1[:, None], x0[None, :], c]
                I11 = img[y1[:, None], x1[None, :], c]
                
                # Bilinear interpolation formula
                resized_channel = (
                    (1 - wy) * ((1 - wx) * I00 + wx * I01) +
                    wy * ((1 - wx) * I10 + wx * I11)
                )
                resized_img[:, :, c] = resized_channel
        
        elif method == 'cubic':
            # Bicubic interpolation (simplified version)
            warnings.warn("Cubic interpolation is a simplified implementation. "
                         "For production use, consider using scipy or OpenCV.")
            # We'll use a simplified approach with linear for demonstration
            # In practice, you might want to use scipy.ndimage.zoom or cv2.resize
            x = np.linspace(0, cur_width - 1, resized_width)
            y = np.linspace(0, cur_height - 1, resized_height)
            
            resized_img = np.zeros((resized_height, resized_width, channels), dtype=images.dtype)
            
            for c in range(channels):
                # Simple cubic-like interpolation (catmull-rom)
                from scipy.interpolate import RectBivariateSpline
                if channels == 1:
                    interpolator = RectBivariateSpline(
                        np.arange(cur_height), 
                        np.arange(cur_width), 
                        img[:, :, c], 
                        kx=3, ky=3
                    )
                else:
                    interpolator = RectBivariateSpline(
                        np.arange(cur_height), 
                        np.arange(cur_width), 
                        img[:, :, c], 
                        kx=3, ky=3
                    )
                resized_img[:, :, c] = interpolator(y, x)
        
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'linear', 'nearest', or 'cubic'.")
        
        resized_images.append(resized_img)
    
    resized_images = np.array(resized_images)
    
    # Convert back to original dtype with appropriate clipping
    if original_dtype == np.uint8:
        resized_images = np.round(resized_images).clip(0, 255).astype(np.uint8)
    elif original_dtype == np.float32:
        resized_images = resized_images.clip(-1.0, 1.0).astype(np.float32)
    else:
        raise ValueError(f"Unsupported image dtype: {original_dtype}")
    
    # Calculate padding amounts (centered padding)
    pad_h = height - resized_height
    pad_w = width - resized_width
    
    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    
    # Apply padding
    if original_dtype == np.uint8:
        pad_value = 0
    else:  # float32
        pad_value = -1.0
    
    padded_images = np.pad(
        resized_images,
        pad_width=(
            (0, 0),  # No padding on batch dimension
            (pad_h_top, pad_h_bottom),  # Padding on height
            (pad_w_left, pad_w_right),  # Padding on width
            (0, 0)   # No padding on channel dimension
        ),
        mode='constant',
        constant_values=pad_value
    )
    
    # Remove batch dimension if input didn't have one
    if not has_batch_dim:
        padded_images = padded_images[0]
    
    return padded_images


# Alternative version using OpenCV (if available) for better performance:
def resize_with_pad_cv2(
    images: np.ndarray,
    height: int,
    width: int,
    interpolation: int = None
) -> np.ndarray:
    """Version using OpenCV for resizing (requires opencv-python)."""
    try:
        import cv2
    except ImportError:
        raise ImportError("OpenCV is required for this function. Install with: pip install opencv-python")
    
    original_shape = images.shape
    original_dtype = images.dtype
    has_batch_dim = images.ndim == 4
    
    if not has_batch_dim:
        images = images[None, ...]
    
    batch_size, cur_height, cur_width, channels = images.shape
    
    # Calculate resize ratio
    width_ratio = cur_width / width
    height_ratio = cur_height / height
    ratio = max(width_ratio, height_ratio)
    
    # Calculate new dimensions
    resized_height = int(np.round(cur_height / ratio))
    resized_width = int(np.round(cur_width / ratio))
    
    # Ensure minimum dimensions
    resized_height = max(1, resized_height)
    resized_width = max(1, resized_width)
    
    # Map interpolation method
    if interpolation is None:
        interpolation = cv2.INTER_LINEAR
    
    # Resize each image
    resized_images = []
    for i in range(batch_size):
        img = images[i]
        # OpenCV uses BGR by default, need to handle RGB properly
        if channels == 3:
            # Convert RGB to BGR for OpenCV, then back
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            resized_bgr = cv2.resize(
                img_bgr, 
                (resized_width, resized_height), 
                interpolation=interpolation
            )
            resized_img = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Grayscale or other channel counts
            resized_img = cv2.resize(
                img, 
                (resized_width, resized_height), 
                interpolation=interpolation
            )
            if resized_img.ndim == 2 and channels > 1:
                resized_img = resized_img[..., None]
        
        resized_images.append(resized_img)
    
    resized_images = np.array(resized_images)
    
    # Convert back to original dtype
    if original_dtype == np.uint8:
        resized_images = np.round(resized_images).clip(0, 255).astype(np.uint8)
    elif original_dtype == np.float32:
        resized_images = resized_images.clip(-1.0, 1.0).astype(np.float32)
    
    # Calculate and apply padding
    pad_h = height - resized_height
    pad_w = width - resized_width
    
    pad_h_top = pad_h // 2
    pad_h_bottom = pad_h - pad_h_top
    pad_w_left = pad_w // 2
    pad_w_right = pad_w - pad_w_left
    
    pad_value = 0 if original_dtype == np.uint8 else -1.0
    
    padded_images = np.pad(
        resized_images,
        pad_width=(
            (0, 0),
            (pad_h_top, pad_h_bottom),
            (pad_w_left, pad_w_right),
            (0, 0)
        ),
        mode='constant',
        constant_values=pad_value
    )
    
    if not has_batch_dim:
        padded_images = padded_images[0]
    
    return padded_images


# Example usage:
if __name__ == "__main__":
    # Create a test image
    test_image = np.random.randint(0, 256, size=(100, 150, 3), dtype=np.uint8)
    
    # Resize with padding
    resized = resize_with_pad(test_image, height=200, width=200, method='linear')
    
    print(f"Original shape: {test_image.shape}")
    print(f"Resized shape: {resized.shape}")
    
    # Test with float image
    test_float = np.random.uniform(-1, 1, size=(50, 75, 1)).astype(np.float32)
    resized_float = resize_with_pad(test_float, height=100, width=100, method='linear')
    
    print(f"\nFloat original shape: {test_float.shape}")
    print(f"Float resized shape: {resized_float.shape}")
    print(f"Float value range: [{resized_float.min():.3f}, {resized_float.max():.3f}]")
