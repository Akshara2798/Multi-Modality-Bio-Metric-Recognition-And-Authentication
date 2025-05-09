import numpy as np 
from scipy.stats import kendalltau
from scipy.spatial.distance import braycurtis
from Existing import *
from scipy.spatial.distance import cdist
from keras.models import load_model
from Optimization import *
from Sim_OpPTr import *
from __pycache__.utils import *
import matplotlib.font_manager as font_manager
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from absl import logging as absl_logging
absl_logging.set_verbosity(absl_logging.ERROR)



def Metrix(feature_1,feature_2): 
    
    #Pearson corelation coefficient
    covariance = np.cov(feature_1, feature_2, ddof=0)[0, 1]
    std_x = np.std(feature_1)
    std_y = np.std(feature_2)
    correlation = covariance / (std_x * std_y)
    x = np.array(feature_1)
    y = np.array(feature_2)
    # Euclidean Distance
    distance = np.sqrt(np.sum((x - y) ** 2))
   
    # Manhattan Distance
    Manhattan_distance = np.sum(np.abs(x - y))
    
    # Spearman Rank Correlation Coefficient
    rank_x = np.argsort(np.argsort(feature_1))
    rank_y = np.argsort(np.argsort(feature_2))
    d = rank_x - rank_y
    n = len(x)
    Spear_coefficient = 1 - (6 * np.sum(d ** 2)) / (n * (n ** 2 - 1))
    tau, p_value = kendalltau(feature_1, feature_2)
    # Bray-Curtis dissimilarity
    dissimilarity = braycurtis(feature_1, feature_2)
   
    
    return [correlation,distance,Manhattan_distance,Spear_coefficient,tau,dissimilarity]

def simi(features,Database):
    threshold = 0.95 

    distances = cdist(features, Database, metric='cosine')

    similarity = 1 - distances

    matches = []

    for i, similarities in enumerate(similarity):
        for j, sim in enumerate(similarities):
            if sim > threshold:
                matches.append((i, j, sim))  

   
    similarities_,f_ind,db_ind=[],[],[]
    for match in matches:
        feature_index, db_index, sim = match
        f_ind.append(feature_index)
        similarities_.append(sim)
        db_ind.append(db_index)
   
        
    max_sim=max(similarities_)
    sim_i_feature=f_ind[similarities_.index(max_sim)]
    feature_1=features[sim_i_feature]
    feature_2=db_ind[similarities_.index(max_sim)]
    feature_2=Database[feature_2]
    return [feature_1,feature_2,max_sim]

def Deep_CNN(fused_features,Database):
   
    fused_features=fused_features.reshape(fused_features.shape[0],fused_features.shape[-1],1)
    Deep_CNN = load_model("Model/Deep_CNN_model.h5")
    deep_cnn_pred=Deep_CNN.predict(fused_features)
    Database=np.load("Features/Database_DC.npy")
    features=deep_cnn_pred.reshape(-1,deep_cnn_pred.shape[-1])
    Database=Database.reshape(-1, Database.shape[-1])
    features= simi(features,Database)
    return features

def Bi_Gru(fused_features,Database):  
    fused_features=fused_features.reshape(fused_features.shape[0], 1, fused_features.shape[-1])
    bigru = load_model("Model/Bi_GRU.h5")
    bigru_pred=bigru.predict(fused_features)
    Database=np.load("Features/Database_BiG.npy")
    features= simi(bigru_pred,Database)
    return features
    
def Auto_encoder(fused_features,Database):      
    fused_features=fused_features.reshape(fused_features.shape[0], 1, fused_features.shape[-1])
    auto_encode = load_model("Model/Auto_Encoder.h5")
    ae_pred=auto_encode.predict(fused_features)
    Database=np.load("Features/Database_AE.npy")
    features= simi(ae_pred,Database)
    return features
    
def Attn_Capsule(fused_features,Database):

    fused_features=fused_features.reshape(fused_features.shape[0], 1, fused_features.shape[-1])
    custom_objects = {
            'CapsuleLayer': CapsuleLayer
            
        }
    atten_caps=load_model("Model/Attn_Capsule_Net.h5",custom_objects=custom_objects)
    atten_caps_pred=atten_caps.predict(fused_features)
    Database=np.load("Features/Database_ATC.npy")
    features=atten_caps_pred.reshape(-1,atten_caps_pred.shape[-1])
    Database=Database.reshape(-1, Database.shape[-1])
    features= simi(features,Database)
    return features


def Bilstm_pred(fused_features,Database):
    bilstm_model=load_model("Model/AbBilstm.h5")
    
    fused_features=fused_features.reshape(fused_features.shape[0],1,fused_features.shape[-1])
    bilstm_model_pred=bilstm_model.predict(fused_features)
    Database=np.load("Features/Bilstm_DB.npy")
    Database=Database.reshape(Database.shape[0],Database.shape[-1])
    feature=bilstm_model_pred.reshape(bilstm_model_pred.shape[0],bilstm_model_pred.shape[-1])
    features=simi(feature,Database)
    return features


def ta_pred(fused_features,Database):
    model=load_model("Model/TEA_model.h5")
    Database=np.load("Features/TEA_DB.npy")
    fused_features=fused_features.reshape(fused_features.shape[0],1,fused_features.shape[-1])[:Database.shape[-1]]
    pred_f=model.predict(fused_features)
    Database=Database.reshape(Database.shape[0],Database.shape[-1])
    feature=pred_f.reshape(pred_f.shape[0],pred_f.shape[-1])
    features=simi(feature,Database)
    return features

def bilstm_atten_pred(fused_features,Database):
    model=load_model("Model/TEA_model.h5")
    Database=np.load("Features/TEA_DB.npy")
    fused_features=fused_features.reshape(fused_features.shape[0],1,fused_features.shape[-1])[:Database.shape[-1]]
    pred_f=model.predict(fused_features)
    Database=Database.reshape(Database.shape[0],Database.shape[-1])
    feature=pred_f.reshape(pred_f.shape[0],pred_f.shape[-1])
    features=simi(feature,Database)
    return features


def Pooling_Atten_trains(fused_features,Database):
    model=load_model("Model/TEA_model.h5")
    Database=np.load("Features/TEA_DB.npy")
    fused_features=fused_features.reshape(fused_features.shape[0],1,fused_features.shape[-1])[:Database.shape[-1]]
    pred_f=model.predict(fused_features)
    Database=Database.reshape(Database.shape[0],Database.shape[-1])
    feature=pred_f.reshape(pred_f.shape[0],pred_f.shape[-1])
    features=simi(feature,Database)
    return features
    
    
    
def Plot(feature_1,feature_2,fused_features,Database,max_sim):
    proposed_performance=Metrix(feature_1,feature_2)
    Accuracy_proposed=max_sim
    Accuracy_proposed=float(Accuracy_proposed)
    pr_cof_proposed=proposed_performance[0]
    pr_cof_proposed=float(pr_cof_proposed)
    Euclidean_proposed=proposed_performance[1]
    Euclidean_proposed=float(Euclidean_proposed)
    Manhatt_proposed=proposed_performance[2]
    Manhatt_proposed=float(Manhatt_proposed)
    Spear_proposed=proposed_performance[3]
    Spear_proposed=float(Spear_proposed)
    Kendall_proposed=proposed_performance[4]
    Kendall_proposed=float(Kendall_proposed)
    Bray_Curtis_proposed=proposed_performance[5]
    Bray_Curtis_proposed=float(Bray_Curtis_proposed)
    
    print()
    print("Proposed Performance : \n********************\n")
    print("Accuracy                           :",Accuracy_proposed)
    print("Pearson Correlation Coefficient    :",pr_cof_proposed)
    print('Euclidean Distance                 :',Euclidean_proposed)
    print("Manhattan Distance                 :",Manhatt_proposed) 
    print("Spearman Rank Correlation          :",Spear_proposed)
    print("Kendall Tau Coefficient            :",Kendall_proposed) 
    print("Bray-Curtis dissimilarity          :",Bray_Curtis_proposed)

    
    
    
    deepcnn=Deep_CNN(fused_features,Database)
    deepcnn_performance=Metrix(deepcnn[0],deepcnn[1])
    deepcnn_accuracy=deepcnn[-1]
    deepcnn_accuracy=float(deepcnn_accuracy)
    pr_cof_deepcnn=deepcnn_performance[0]
    pr_cof_deepcnn=float(pr_cof_deepcnn)
    Euclidean_deepcnn=deepcnn_performance[1]
    Euclidean_deepcnn=float(Euclidean_deepcnn)
    Manhatt_deepcnn=deepcnn_performance[2]
    Manhatt_deepcnn=float(Manhatt_deepcnn)
    Spear_deepcnn=deepcnn_performance[3]
    Spear_deepcnn=float(Spear_deepcnn)
    Kendall_deepcnn=deepcnn_performance[4]
    Kendall_deepcnn=float(Kendall_deepcnn)
    Bray_Curtis_deepcnn=deepcnn_performance[5]
    Bray_Curtis_deepcnn=float(Bray_Curtis_deepcnn)
    
    
    print()
    print("Deep CNN Performance : \n********************\n")
    print("Accuracy                           :",deepcnn_accuracy)
    print("Pearson Correlation Coefficient    :",pr_cof_deepcnn)
    print('Euclidean Distance                 :',Euclidean_deepcnn)
    print("Manhattan Distance                 :",Manhatt_deepcnn) 
    print("Spearman Rank Correlation          :",Spear_deepcnn)
    print("Kendall Tau Coefficient            :",Kendall_deepcnn) 
    print("Bray-Curtis dissimilarity          :",Bray_Curtis_deepcnn)

    
    bigru_=Bi_Gru(fused_features,Database)
    bigru_performance=Metrix(bigru_[0],bigru_[1])
    bigru_accuracy=bigru_[-1]
    bigru_accuracy=float(bigru_accuracy)
    pr_cof_bigru=bigru_performance[0]
    pr_cof_bigru=float(pr_cof_bigru)
    Euclidean_bigru=bigru_performance[1]
    Euclidean_bigru=float(Euclidean_bigru)
    Manhatt_bigru=bigru_performance[2]
    Manhatt_bigru=float(Manhatt_bigru)
    Spear_bigru=bigru_performance[3]
    Spear_bigru=float(Spear_bigru)
    Kendall_bigru=bigru_performance[4]
    Kendall_bigru=float(Kendall_bigru)
    Bray_Curtis_bigru=bigru_performance[5]
    Bray_Curtis_bigru=float(Bray_Curtis_bigru)
    
    
    print()
    print("Bi Gru Performance : \n********************\n")
    print("Accuracy                           :",bigru_accuracy)
    print("Pearson Correlation Coefficient    :",pr_cof_bigru)
    print('Euclidean Distance                 :',Euclidean_bigru)
    print("Manhattan Distance                 :",Manhatt_bigru) 
    print("Spearman Rank Correlation          :",Spear_bigru)
    print("Kendall Tau Coefficient            :",Kendall_bigru) 
    print("Bray-Curtis dissimilarity          :",Bray_Curtis_bigru)
    
    
    
    ae_=Auto_encoder(fused_features,Database)
    ae_performance=Metrix(ae_[0],ae_[1])
    ae_accuracy=ae_[-1]
    ae_accuracy=float(ae_accuracy)
    pr_cof_ae=ae_performance[0]
    pr_cof_ae=float(pr_cof_ae)
    Euclidean_ae=ae_performance[1]
    Euclidean_ae=float(Euclidean_ae)
    Manhatt_ae=ae_performance[2]
    Manhatt_ae=float(Manhatt_ae)
    Spear_ae=ae_performance[3]
    Spear_ae=float(Spear_ae)
    Kendall_ae=ae_performance[4]
    Kendall_ae=float(Kendall_ae)
    Bray_Curtis_ae=ae_performance[5]
    Bray_Curtis_ae=float(Bray_Curtis_ae)
    
    print()
    print("Auto Encoder Performance : \n********************\n")
    print("Accuracy                           :",ae_accuracy)
    print("Pearson Correlation Coefficient    :",pr_cof_ae)
    print('Euclidean Distance                 :',Euclidean_ae)
    print("Manhattan Distance                 :",Manhatt_ae) 
    print("Spearman Rank Correlation          :",Spear_ae)
    print("Kendall Tau Coefficient            :",Kendall_ae) 
    print("Bray-Curtis dissimilarity          :",Bray_Curtis_ae)
    
    
    attn_caps=Attn_Capsule(fused_features,Database)
    attn_caps_performance=Metrix(attn_caps[0],attn_caps[1])
    attn_caps_accuracy=attn_caps[-1]
    attn_caps_accuracy=float(attn_caps_accuracy)
    pr_cof_attn_caps=ae_performance[0]
    pr_cof_attn_caps=float(pr_cof_attn_caps)
    Euclidean_attn_caps=attn_caps_performance[1]
    Euclidean_attn_caps=float(Euclidean_attn_caps)
    Manhatt_attn_caps=attn_caps_performance[2]
    Manhatt_attn_caps=float(Manhatt_attn_caps)
    Spear_attn_caps=attn_caps_performance[3]
    Spear_attn_caps=float(Spear_attn_caps)
    Kendall_attn_caps=attn_caps_performance[4]
    Kendall_attn_caps=float(Kendall_attn_caps)
    Bray_Curtis_attn_caps=attn_caps_performance[5]
    Bray_Curtis_attn_caps=float(Bray_Curtis_attn_caps)
    
    print()
    print("Attention Capsule Network Performance : \n********************\n")
    print("Accuracy                           :",attn_caps_accuracy)
    print("Pearson Correlation Coefficient    :",pr_cof_attn_caps)
    print('Euclidean Distance                 :',Euclidean_attn_caps)
    print("Manhattan Distance                 :",Manhatt_attn_caps) 
    print("Spearman Rank Correlation          :",Spear_attn_caps)
    print("Kendall Tau Coefficient            :",Kendall_attn_caps) 
    print("Bray-Curtis dissimilarity          :",Bray_Curtis_attn_caps)
    
    
    
    # Ablation_study
    
    bilstm=Bilstm_pred(fused_features,Database)
    bilstm_accuracy=bilstm[-1]
    bilstm_accuracy=float(bilstm_accuracy)
    tea=ta_pred(fused_features,Database)
    tea_accuracy=tea[-1]
    tea_accuracy=float(tea_accuracy)
    
    bilstm_attn=bilstm_atten_pred(fused_features,Database)
    bilstm_attn_accuracy=bilstm_attn[-1]
    bilstm_attn_accuracy=float(bilstm_attn_accuracy)
    print(bilstm_attn)
    pool_atten_trans=Pooling_Atten_trains(fused_features,Database)
    print(pool_atten_trans)
    pool_atten_trans_accuracy=pool_atten_trans[-1]
    pool_atten_trans_accuracy=float(pool_atten_trans_accuracy)
    
    font = font_manager.FontProperties(
        family='Times New Roman', style='normal', size=14, weight='bold')
    
   
    legend_properties = {'weight': 'bold', 'family': 'Times New Roman', 'size': 14}
    
    con = "Deep CNN"
    con1 = "Bi-GRU" 
    con2 = "AE"
    con3 = "Atten-Caps"
    con4="Proposed"
     
    
    plt.figure()
    width = 0.25
    plt.bar(0,deepcnn_accuracy*100, width, color='r', align='center', edgecolor='black',) 
    plt.bar(1,bigru_accuracy*100, width, color='g', align='center', edgecolor='black') 
    plt.bar(2,ae_accuracy*100, width, color='#68228B', align='center', edgecolor='black') 
    plt.bar(3,attn_caps_accuracy*100, width, color='#800000', align='center', edgecolor='black') 
    plt.bar(4,Accuracy_proposed*100, width, color='#0000FF', align='center', edgecolor='black') 
       
    
    plt.xticks(np.arange(5),(con, con1,con2,con3,con4),fontweight='bold',fontsize=14,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('Accuracy (%)',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Result/Accuracy.png', format="png",dpi=600)
    
    
    plt.figure()
    width = 0.25
    plt.bar(0,pr_cof_deepcnn, width, color='r', align='center', edgecolor='black',) 
    plt.bar(1,pr_cof_bigru, width, color='g', align='center', edgecolor='black') 
    plt.bar(2,pr_cof_ae, width, color='#68228B', align='center', edgecolor='black') 
    plt.bar(3,pr_cof_attn_caps, width, color='#800000', align='center', edgecolor='black') 
    plt.bar(4,pr_cof_proposed, width, color='#0000FF', align='center', edgecolor='black') 
       
    
    plt.xticks(np.arange(5),(con, con1,con2,con3,con4),fontweight='bold',fontsize=14,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('Pearson corelation coefficient',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Result/Pearson corelation coefficient.png', format="png",dpi=600)
    
    
    plt.figure()
    width = 0.25
    plt.bar(0,Euclidean_deepcnn, width, color='r', align='center', edgecolor='black',) 
    plt.bar(1,Euclidean_bigru, width, color='g', align='center', edgecolor='black') 
    plt.bar(2,Euclidean_ae, width, color='#68228B', align='center', edgecolor='black') 
    plt.bar(3,Euclidean_attn_caps, width, color='#800000', align='center', edgecolor='black') 
    plt.bar(4,Euclidean_proposed, width, color='#0000FF', align='center', edgecolor='black') 
       
    
    plt.xticks(np.arange(5),(con, con1,con2,con3,con4),fontweight='bold',fontsize=14,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('Euclidean Distance',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Result/Euclidean Distance.png', format="png",dpi=600)
    
    
    plt.figure()
    width = 0.25
    plt.bar(0,Manhatt_deepcnn, width, color='r', align='center', edgecolor='black',) 
    plt.bar(1,Manhatt_bigru, width, color='g', align='center', edgecolor='black') 
    plt.bar(2,Manhatt_ae, width, color='#68228B', align='center', edgecolor='black') 
    plt.bar(3,Manhatt_attn_caps, width, color='#800000', align='center', edgecolor='black') 
    plt.bar(4,Manhatt_proposed, width, color='#0000FF', align='center', edgecolor='black') 
       
    
    plt.xticks(np.arange(5),(con, con1,con2,con3,con4),fontweight='bold',fontsize=14,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('Manhattan Distance',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Result/Manhattan Distance.png', format="png",dpi=600)
    
    
    plt.figure()
    width = 0.25
    plt.bar(0,Spear_deepcnn, width, color='r', align='center', edgecolor='black',) 
    plt.bar(1,Spear_bigru, width, color='g', align='center', edgecolor='black') 
    plt.bar(2,Spear_ae, width, color='#68228B', align='center', edgecolor='black') 
    plt.bar(3,Spear_attn_caps, width, color='#800000', align='center', edgecolor='black') 
    plt.bar(4,Spear_proposed, width, color='#0000FF', align='center', edgecolor='black') 
       
    
    plt.xticks(np.arange(5),(con, con1,con2,con3,con4),fontweight='bold',fontsize=14,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('Spearman Rank Correlation Coefficient',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Result/Spearman Rank Correlation Coefficient.png', format="png",dpi=600)
    
    
    plt.figure()
    width = 0.25
    plt.bar(0,Kendall_deepcnn, width, color='r', align='center', edgecolor='black',) 
    plt.bar(1,Kendall_bigru, width, color='g', align='center', edgecolor='black') 
    plt.bar(2,Kendall_ae, width, color='#68228B', align='center', edgecolor='black') 
    plt.bar(3,Kendall_attn_caps, width, color='#800000', align='center', edgecolor='black') 
    plt.bar(4,Kendall_proposed, width, color='#0000FF', align='center', edgecolor='black') 
       
    
    plt.xticks(np.arange(5),(con, con1,con2,con3,con4),fontweight='bold',fontsize=14,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('Kendall Tau Coefficient',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Result/Kendall Tau Coefficient.png', format="png",dpi=600)
    
    plt.figure()
    width = 0.25
    plt.bar(0,Bray_Curtis_deepcnn, width, color='r', align='center', edgecolor='black',) 
    plt.bar(1,Bray_Curtis_bigru, width, color='g', align='center', edgecolor='black') 
    plt.bar(2,Bray_Curtis_ae, width, color='#68228B', align='center', edgecolor='black') 
    plt.bar(3,Bray_Curtis_attn_caps, width, color='#800000', align='center', edgecolor='black') 
    plt.bar(4,Bray_Curtis_proposed, width, color='#0000FF', align='center', edgecolor='black') 
       
    
    plt.xticks(np.arange(5),(con, con1,con2,con3,con4),fontweight='bold',fontsize=14,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('Bray-Curtis dissimilarity',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Result/Bray-Curtis dissimilarity.png', format="png",dpi=600)
    
    
    con = "Module-1"
    con1 = "Module-2" 
    con2 = "Module-3"
    con3 = "Module-4"
    con4="Module-5\n(Proposed)"
    
     
    print(bilstm_accuracy*100)
    print(tea_accuracy*100)
    print(bilstm_attn_accuracy*100)
    print(pool_atten_trans_accuracy*100)
    print(Accuracy_proposed*100)
    plt.figure()
    plt.ylim(30,100)
    width = 0.25
    plt.bar(0,bilstm_accuracy*100, width, color='r', align='center', edgecolor='black',) 
    plt.bar(1,tea_accuracy*100, width, color='g', align='center', edgecolor='black') 
    plt.bar(2,bilstm_attn_accuracy*100, width, color='#68228B', align='center', edgecolor='black') 
    plt.bar(3,pool_atten_trans_accuracy*100, width, color='#800000', align='center', edgecolor='black') 
    plt.bar(4,Accuracy_proposed*100, width, color='#0000FF', align='center', edgecolor='black') 
       
    
    plt.xticks(np.arange(5),(con, con1,con2,con3,con4),fontweight='bold',fontsize=14,fontname = "Times New Roman")
    plt.yticks( fontweight='bold',fontsize=16,fontname = "Times New Roman")
    plt.ylabel('Accuracy (%)',fontweight='bold',fontsize=18,fontname = "Times New Roman")
    plt.grid(linestyle='--', linewidth=0.2)   
    plt.title(" ",fontweight='bold',fontsize=18)
    plt.savefig('Result/Ablation_study.png', format="png",dpi=600)