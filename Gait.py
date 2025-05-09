import os
import cv2
import torch
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn
from PIL import Image
import numpy as np

# Load a pretrained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

def process_image(image_path, gait_images_folder):
    # Ensure the gait images folder exists
    os.makedirs(gait_images_folder, exist_ok=True)

    # Extract filename from the image path
    filename = os.path.basename(image_path)

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    # Define a preprocessing pipeline that converts the image to a tensor
    preprocess = T.Compose([T.ToTensor()])
    
    # Apply the preprocessing pipeline to the image (convert it to a tensor)
    input_tensor = preprocess(image)
    
    # Add an extra batch dimension (since the model expects a batch of images)
    input_batch = input_tensor.unsqueeze(0)
    
    # Run the Mask R-CNN model to make predictions, with no gradient computation (inference mode)
    with torch.no_grad():
        predictions = model(input_batch)
    
    # Extract the masks predicted by the model (assuming it's the first item in predictions)
    masks = predictions[0]['masks']
    
    # Check if there is at least one mask in the result
    if len(masks) > 0:  
        # Get the first mask and move it to CPU (from GPU) for processing
        mask = masks[0, 0].cpu().numpy()
    
        # Normalize the mask for saving (remove extra dimensions and convert values to a suitable range)
        mask = np.squeeze(mask)  # Remove single-dimensional entries
        mask = (mask * 255).astype(np.uint8)  # Convert mask to 8-bit grayscale image
    
        # Define the path where the mask image will be saved
        gait_image_path = os.path.join(gait_images_folder, filename)
        
        # Save the mask image as a grayscale image
        cv2.imwrite(gait_image_path, mask)


def test_process(image):
     preprocess = T.Compose([T.ToTensor()])
     input_tensor = preprocess(image)
     input_batch = input_tensor.unsqueeze(0)

     # Run the Mask R-CNN model
     with torch.no_grad():
         predictions = model(input_batch)

     # Extract masks
     masks = predictions[0]['masks']
     if len(masks) > 0:  # Ensure there is at least one mask
         mask = masks[0, 0].cpu().numpy()

         # Normalize mask for saving
         mask = np.squeeze(mask)  # Remove single-dimensional entries
         mask = (mask * 255).astype(np.uint8)  # Convert mask to 8-bit grayscale image


         return mask
     
