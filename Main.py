import numpy as np
import cv2
import os
from Preprocess import tpdmf
from key_frames_select import initialize_clusters, calculate_optimal_k, merge_clusters, select_key_frames
from Feature_extraction import PoC_Den
from Sim_OpPTr import Sim_OpPTr
from Existing import Deep_CNN,Bi_GRU,Auto_encoder,Attn_capsule_network
from Sim_OpPTr import PoolingAttention
from Optimization import Sim_GHO_Optimization
from ultralytics import YOLO
from PIL import Image
from Gait import process_image 
from keras.models import load_model 


def Base():
    # Function to extract frames from a video
    def extract_frames(video_path, interval=10):
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
    
        while True:
            ret, frame = cap.read()
            if not ret:
                break
    
            # Capture frame at the specified interval
            if frame_count % interval == 0:
                frames.append(frame)
            frame_count += 1
        len(frames)
        cap.release()
        return frames
    
    # Paths and initialization
    dataset_path = "Dataset/video_1"
    keyframe_dict = {}
    preprocessed_frame_list = []
    Labels = []  # To store labels
    
    # Process the first 10 folders in the dataset
    dataset_dir_list = os.listdir(dataset_path)
    for folder in dataset_dir_list:
        folder_path =dataset_path+"/"+folder 
        
        video_path = folder_path
        print(f"Processing video: {video_path}")
    
        # Extract frames from the video
        Extracted_frames = extract_frames(video_path)
        
        # Key frame selection
        print("---------------Key frame selection---------------------")
        '''Enhanced Agglomerative Nesting clustering algorithm (EAg-NCA)'''
        initial_clusters = initialize_clusters(Extracted_frames)
        optimal_k = calculate_optimal_k(initial_clusters)
        final_clusters = merge_clusters(initial_clusters, optimal_k)
        key_frames = select_key_frames(final_clusters, Extracted_frames)
    
        # Save keyframes
        key_frame_path = 'selected Frames/'+video_path.split('.')[0].split("/")[-1]
        os.makedirs(key_frame_path, exist_ok=True)
    
        keyframe_dict[video_path] = []
        for idx, key_frame in enumerate(key_frames):
            frame_filename = f"Image_{idx}.png"
            key_frame_file_path = os.path.join(key_frame_path, frame_filename)
            cv2.imwrite(key_frame_file_path, key_frame)
            keyframe_dict[video_path].append(key_frame_file_path)
            
            # Append label based on folder name
            Labels.append(folder)
            # np.save('Files/Labels.npy', Labels)
    
    
    
    
    for video_path, keyframe_files in keyframe_dict.items():
        preprocessed_frames = []
        for keyframe_file_path in keyframe_files:
            print(f"Preprocessing key frame: {keyframe_file_path}")
            key_frame_image = cv2.imread(keyframe_file_path)
    
            # Resize the image to (350,350)
            resized_frame = cv2.resize(key_frame_image, (350,350))
            
            "-----------------------------------Preprocess-----------------------------------------"
            # Apply the TPDMF preprocessing with window_size=3
            '''Trimmed Pixel density based median filter (TPDMF)'''
            preprocessed_frame = tpdmf(resized_frame, window_size=3, trim_ratio=0.2)
            preprocessed_frames.append(preprocessed_frame)
            
            # Save the preprocessed frame
            preprocessed_file_path = keyframe_file_path.replace('selected Frames', 'Preprocessed_frames')
            os.makedirs(os.path.dirname(preprocessed_file_path), exist_ok=True)
            cv2.imwrite(preprocessed_file_path, preprocessed_frame)
    
        preprocessed_frame_list.append(preprocessed_frames)
    
    print("Preprocessing of keyframes completed.")
    
    pre_dir="Preprocessed_frames"
    
    for folders in os.listdir(pre_dir):
        for i in os.listdir(pre_dir+"/"+folders):
            img=cv2.imread(pre_dir+"/"+folders+"/"+i)
            print("readed")
            cv2.imwrite("Features/Preprocessed_images/"+folders+"_"+i,img)
    
    # np.save("Features/key_frames.npy",key_frames)
            
    image_folders = [
        "Features/Preprocessed_images",
    ]
    results_folder = "results"
    values_folder = "values"
    face_image_folder = "face_image"
    gait_images_folder = "gait_images"
    
    # Create necessary folders if they don't exist
    os.makedirs(results_folder, exist_ok=True)
    os.makedirs(values_folder, exist_ok=True)
    os.makedirs(face_image_folder, exist_ok=True)
    os.makedirs(gait_images_folder, exist_ok=True)
    "------------------------------------------Face Detection -----------------------------------------"
    '''Yolo v9 Dung Beetle Optimization tune the parameters'''
    
    yolo_model = YOLO('Model/YoloV9.pt')
    all_image_filenames = set()
    for folder in image_folders:
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_image_filenames.add(filename.lower())
    
    
    face_images_list = []
    gait_images_list = []
    
    
    for folder in image_folders:
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(folder, filename)
    
                # Perform YOLO inference for face detection
                results = yolo_model.predict(source=image_path)
    
                # Save the output images with bounding boxes and crop faces
                for result in results:
                    output_image_path = os.path.join(results_folder, f"{os.path.splitext(filename)[0]}.png")
                    result.save(output_image_path)
    
                    # Save bounding box details to a .txt file
                    txt_filename = f"{os.path.splitext(filename)[0]}.txt"
                    txt_path = os.path.join(values_folder, txt_filename)
    
                    with open(txt_path, 'w') as f:
                        for box in result.boxes:
                            # Extract box details
                            x1, y1, x2, y2 = box.xyxy[0].tolist()  # bounding box coordinates
                            conf = box.conf[0].item()  # confidence score
                            cls = int(box.cls[0].item())  # class label
                            # Write details to the file
                            f.write(f'{cls}, {x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}\n')
    
                    # Crop, resize, and save face images immediately after saving the result
                    img = Image.open(output_image_path)
                    img_width, img_height = img.size
    
                    with open(txt_path, 'r') as f:
                        lines = f.readlines()
    
                    for idx, line in enumerate(lines):
                        parts = line.strip().split(',')
                        if len(parts) != 5:
                            print(f"Invalid format in {txt_filename} on line {idx+1}")
                            continue
    
                        cls, x1, y1, x2, y2 = map(float, parts)
    
                        # Ensure coordinates are within image bounds
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(img_width, x2), min(img_height, y2)
    
                        # Crop the face from the image
                        cropped_img = img.crop((x1, y1, x2, y2))
    
                        # Resize the cropped image
                        resized_img = cropped_img.resize((80, 80), Image.LANCZOS)
    
                        # Convert resized image to NumPy array and append to list
                        face_images_list.append(np.array(resized_img))
    
                        # Define the output path for the resized face image
                        face_image_path = os.path.join(face_image_folder, filename)
                        resized_img.save(face_image_path)
    
                        # Check if the face image filename is present in image_folders
                        if filename.lower() in all_image_filenames:
                            # Find which folder contains the original image
                            original_image_path = None
                            for folder in image_folders:
                                if os.path.exists(os.path.join(folder, filename)):
                                    original_image_path = os.path.join(folder, filename)
                                    break
                            
                            if original_image_path:
                                # Process the original image from the appropriate folder
                                '-----------------------------------Gait Detection-----------------------------------'
                                '''Mask R-CNN model'''
                                process_image(original_image_path, gait_images_folder)
    
    
    imgs=[]
    img_path="Detected_images"
    img_l=os.listdir(img_path)
    for img in img_l:
        img_=cv2.imread(img_path+"/"+img)
        imgs.append(img_)
        
        
    # Save gait images to .npy file
    face_img_path="face_image/"
    face_img_list=[]
    gait_images_list = []
    for filename in os.listdir(gait_images_folder):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            face_path=face_img_path+filename
            img_path="Detected_images"+"/"+filename
            img_=cv2.imread(img_path)
            imgs.append(img_)
            gait_image_path = os.path.join(gait_images_folder, filename)
            gait_img = Image.open(gait_image_path)
            face_img = Image.open(face_path)
            face_img_np = np.array(face_img)
            face_img_list.append(face_img_np)
            gait_img_np = np.array(gait_img)
            gait_images_list.append(gait_img_np)
    # np.save("Features/Gait_images.npy", gait_images_list)
    # np.save("Features/Face_images.npy",face_img_list)
    # np.save("Features/detected_imgs.npy",imgs)
    
    
    '--------------------- Feature Extraction------------------'
    '''pooled convolutional dense net model (PoC-Den) '''
    
    face_image = np.load('Features/detected_imgs.npy')
    face_PoC_Den_model = PoC_Den(face_image)
    
    face_PoC_Den_model=load_model("Model/face_model12.h5")
    face_features=face_PoC_Den_model.predict(face_image)
    
    gait_image = np.load('Features/Gait_images.npy')
    gait_image = np.expand_dims(gait_image, axis=-1)
    gait_PoC_Den_model = PoC_Den(gait_image)
    
    gait_PoC_Den_model=load_model("Model/Gait_model12.h5")
    gait_features=gait_PoC_Den_model.predict(gait_image)
    
    
    face_features=np.load("Features/detected_features.npy")
    gait_features=np.load("Features/Gait_features.npy")
    # Concatenate features along the last axis
    fused_features = np.concatenate([face_features, gait_features], axis=-1)
    # np.save("Features/Fused_features.npy", fused_features)
    
    
    '----------------------------------Authentication-----------------------------------'
    '''similarity based optimized hybrid bidirectional recurrent neural pooling transformer encoder block (Sim-OpPTr)'''
    fused_features = np.load('Features/Fused_features.npy')
    fused_features = fused_features.reshape(fused_features.shape[0],1,fused_features.shape[-1])
    sim_model_ = Sim_OpPTr(fused_features[0].shape)
    
    
    
    for i in range(len(sim_model_.weights)):
        sim_model_.weights[i]._handle_name = sim_model_.weights[i].name + "_" + str(i)
    
    
    # sim_model_.save("Model/Sim_OpPTr. h5")
    custom_objects = {
            'PoolingAttention': PoolingAttention,
            'Sim_GHO_Optimization':Sim_GHO_Optimization
        }
    proposed_model = load_model("Model/Sim_OpPTr.h5", custom_objects=custom_objects)
    
    features=proposed_model.predict(fused_features)
    # np.save("Features/Database.npy",features)
    
    
    
    '----------------------------Existng-----------------------------'
    # from Existing import *
    
    fused_features_reshaped = fused_features.reshape(fused_features.shape[0], 1, fused_features.shape[-1])
    fused_features=fused_features.reshape(fused_features.shape[0],fused_features.shape[-1],1)
    
    '''Deep CNN'''
    Deep_CNN_model = Deep_CNN(fused_features)
    # Deep_CNN_model.save("Model/Deep_CNN_model.h5")
    
    
    '''Bi GRU'''
    Bi_GRU_model = Bi_GRU(fused_features_reshaped)
    # Bi_GRU_model.save("Model/Bi_GRU.h5")
    
    
    '''Auto Encoder'''
    Auto_Encoder = Auto_encoder(fused_features_reshaped)
    # Auto_Encoder.save("Model/Auto_Encoder.h5")
    
    '''Attention Capsule Network'''
    Attn_Capsule_Network = Attn_capsule_network(fused_features_reshaped)
    # Attn_Capsule_Network.save("Model/Attn_Capsule_Net.h5")
    
