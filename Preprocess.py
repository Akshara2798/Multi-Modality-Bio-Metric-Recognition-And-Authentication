import cv2
import numpy as np

def calculate_density(pixels):
    """
    Calculate the pixel density based on the difference between the pixel and its neighbors.
    """
    center_pixel = pixels[len(pixels) // 2]
    densities = [abs(center_pixel - p) for p in pixels]
    return densities

def tpdmf_channel(channel, window_size=3, trim_ratio=0.2):
    """
    Apply TPDMF to a single channel of the image.

    Parameters:
    channel (numpy.ndarray): Input image channel.
    window_size (int): Size of the neighborhood (must be odd).
    trim_ratio (float): The ratio of pixels to trim from both ends.

    Returns:
    numpy.ndarray: The denoised image channel.
    """
    # Calculate padding size based on the window size (half of window size)
    pad_size = window_size // 2
    
    # Pad the channel with the calculated padding size using border replication
    padded_channel = cv2.copyMakeBorder(channel, pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REPLICATE)
    
    # Initialize an output channel with the same shape as the original channel
    output_channel = np.zeros_like(channel)
    
    # Loop over each pixel in the padded channel (skipping the padding regions)
    for i in range(pad_size, padded_channel.shape[0] - pad_size):
        for j in range(pad_size, padded_channel.shape[1] - pad_size):
            
            # Extract a window around the current pixel and flatten it into a 1D array
            window = padded_channel[i-pad_size:i+pad_size+1, j-pad_size:j+pad_size+1].flatten()
            
            # Calculate the density for each pixel in the window (assuming calculate_density is defined elsewhere)
            densities = calculate_density(window)
            
            # Sort the window pixels by their density values
            pixel_density_pairs = sorted(zip(window, densities), key=lambda x: x[1])
            
            # Calculate the amount of pixels to trim based on the trim ratio
            trim_amount = int(len(window) * trim_ratio)
            
            # Select the pixels in the middle of the sorted list (after trimming the highest and lowest density pixels)
            trimmed_pixels = [pair[0] for pair in pixel_density_pairs[trim_amount:-trim_amount]]
            
            # Set the value of the output pixel to the median of the trimmed pixels
            output_channel[i-pad_size, j-pad_size] = np.median(trimmed_pixels)


    return output_channel

def tpdmf(image, window_size=3, trim_ratio=0.2):
    """
    Apply TPDMF to a color image.

    Parameters:
    image (numpy.ndarray): Input color image.
    window_size (int): Size of the neighborhood (must be odd).
    trim_ratio (float): The ratio of pixels to trim from both ends.

    Returns:
    numpy.ndarray: The denoised color image.
    """
    # Split the image into its color channels
    channels = cv2.split(image)
    
    # Apply TPDMF to each channel
    denoised_channels = [tpdmf_channel(channel, window_size, trim_ratio) for channel in channels]
    
    # Merge the denoised channels back into a color image
    denoised_image = cv2.merge(denoised_channels)
    
    return denoised_image


