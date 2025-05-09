import numpy as np
import cv2
from Preprocess import tpdmf
from key_frames_select import initialize_clusters, calculate_optimal_k, merge_clusters, select_key_frames
from Sim_OpPTr import *
from Optimization import *
from tkinter.filedialog import askopenfilename
import warnings
warnings.filterwarnings('ignore')
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)
from ultralytics import YOLO
from PIL import Image
from Gait import test_process  
from keras.models import load_model 
from scipy.spatial.distance import cdist
import sys
from Performance import Plot


def extract_frames(video_path, interval=10):
    cap = cv2.VideoCapture(video_path)  
    frames = []
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % interval == 0:
            frames.append(frame)
        frame_count += 1

    cap.release()
    return frames

def process_video(video_path):
    print()
    print("Extract Frames...")
    Extracted_frames = extract_frames(video_path)
    "--------------------Key frame selection----------------------"
    print()
    print("Key Frame Selection ...")
    '''Enhanced Agglomerative Nesting clustering algorithm (EAg-NCA)'''
    initial_clusters = initialize_clusters(Extracted_frames)
    optimal_k = calculate_optimal_k(initial_clusters)
    final_clusters = merge_clusters(initial_clusters, optimal_k)
    key_frames = select_key_frames(final_clusters, Extracted_frames)
    #preprocess
    cropped_images_list = []
    detected_img_list=[]
    gait_images_list = []
    detection_found = False
    "-----------------------------------Preprocess-----------------------------------------"
    '''Trimmed Pixel density based median filter (TPDMF)'''
    for idx,key_frame in enumerate(key_frames):
        
        resized_frame = cv2.resize(key_frame, (350,350))
        cv2.imwrite("Result/original/"+"image_"+str(idx)+".jpg", resized_frame)
        preprocessed_frame = tpdmf(resized_frame, window_size=3, trim_ratio=0.2)
        cv2.imwrite("Result/Preprocessed/"+"image_"+str(idx)+".jpg", preprocessed_frame)
        "------------------------------Face Detection ---------------------------------"
        '''Yolo v9 Dung Beetle Optimization tune the parameters'''
        yolo_model = YOLO('Model/YoloV9.pt')
        results = yolo_model.predict(source=preprocessed_frame)
    
        box_processed = False
    
        for result in results:
            if len(result.boxes) > 0:
                detection_found = True
                # Process only the first box detected
                box = result.boxes[0]
                result.show()
                result.save("Result/Detected_image/"+"image_"+str(idx)+".jpg")
                
                # Extract box details
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(350, x2), min(350, y2)
        
                # Convert the image to a PIL Image for cropping
                img = result.orig_img
                pillow_image = Image.fromarray(img)
                detected_img_list.append(img)
               
                # Crop and resize the image
                cropped_img = pillow_image.crop((x1, y1, x2, y2))
                resized_img = cropped_img.resize((80, 80), Image.LANCZOS)
                
                # Append the cropped and resized image to the list
                cropped_images_list.append(np.array(resized_img))
        
                # Set the flag to indicate that a box was processed
                box_processed = True
                
                # Exit the loop after processing the first box
                
        '-----------------------------------Gait Detection-----------------------------------'
        '''Mask R-CNN model'''
        # Process the gait image if a box was processed
        if box_processed:
            gait_image = test_process(preprocessed_frame)
            gait_images = Image.fromarray(gait_image)
            cv2.imwrite("Result/Gait_image/"+"image_"+str(idx)+".jpg",gait_image)
            gait_images.show()
            gait_images_list.append(gait_image)
            detection_found = True
        else:
            pass
        if idx==10:
            break

    return detection_found, cropped_images_list, gait_images_list,detected_img_list
    

def main():
    while True:
        print("Select Input..")
        video_path = askopenfilename(initialdir='Dataset/')
        
        detection_found, cropped_images_list, gait_images_list,d_img = process_video(video_path)
        if not detection_found:
            print("No detection found, restarting...")
            continue
            
            
        cropped_images_array = np.array(cropped_images_list)
        gait_images_array = np.array(gait_images_list)
        d_img=np.array(d_img)
        
        '--------------------- Feature Extraction------------------'
        '''pooled convolutional dense net model (PoC-Den) '''
        
        fe_model=load_model("Model/Feature_ex_model.h5")
        face_PoC_Den_model =load_model("Model/face_model12.h5")
        
        face_features=face_PoC_Den_model.predict(cropped_images_array)
        
        gait_PoC_Den_model = load_model("Model/Gait_model12.h5")
        gait_features=gait_PoC_Den_model.predict(gait_images_array)
        fused_features = np.concatenate([face_features, gait_features], axis=-1)
        
        detc_features=fe_model.predict(d_img)
        fs_f=np.concatenate([detc_features, gait_features], axis=-1)
        fs_f_=fs_f.reshape(fs_f.shape[0],1,fs_f.shape[-1])
        fused_features= fused_features.reshape(fused_features.shape[0],1,fused_features.shape[-1])
        
        
        '----------------------------------Authentication-----------------------------------'
        '''similarity based optimized hybrid bidirectional recurrent neural pooling transformer encoder block (Sim-OpPTr)'''
        
        
        custom_objects = {
                'PoolingAttention': PoolingAttention,
                'Sim_GHO_Optimization':Sim_GHO_Optimization
            }
        proposed_model = load_model("Model/Sim_OpPTr.h5", custom_objects=custom_objects)
        d_f=proposed_model.predict(fs_f_)
        d_f_=d_f.reshape(d_f.shape[0],1,d_f.shape[-1])
        Database=np.load("Features/DB.npy")
        # Define a threshold for similarity
        threshold = 0.99
        
        
        # Ensure Database and features are in the correct shape (2D arrays)
        features = d_f_.reshape(d_f_.shape[0], -1) 
        Database = Database.reshape(Database.shape[0], -1)
        
        
        # Calculate distances
        distances = cdist(features, Database, metric='cosine')
        
        # Convert distances to similarity
        similarity = 1 - distances
        
        matches = []
        
        for i, similarities in enumerate(similarity):
            for j, sim in enumerate(similarities):
                if sim > threshold:
                    matches.append((i, j, sim))  
        
       

        
        similarities_, f_ind, db_ind = [], [], []
        authenticated = False
        authenticated_count = 0
        
        
        for match in matches:
            feature_index, db_index, sim = match
            f_ind.append(feature_index)
            similarities_.append(sim)
            db_ind.append(db_index)
        
        
        for i, similarities in enumerate(similarity):
            max_similarity = np.max(similarities)
            if max_similarity > threshold:
                authenticated_count += 1
                authenticated = True  
            else:
                pass
        if authenticated_count >= 5:
            print("Status: Authenticated")  
        else:
            print("Status: Unauthenticated")  
            sys.exit()  

            
        max_sim=max(similarities_)
        min_sim=min(similarities_)
        sim_i_feature=f_ind[similarities_.index(min_sim)]
        feature_1=features[sim_i_feature]
        feature_2=db_ind[similarities_.index(min_sim)]
        feature_2=Database[feature_2]
        Plot(feature_1,feature_2,fused_features,Database,max_sim)
        
        break
        
if __name__ == "__main__":
    main()