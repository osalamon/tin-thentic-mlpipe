#!/usr/bin/env python3                                                                                 
"""Debug MTCNN face detection with internal thresholds."""                                             
                                                                                                    
from PIL import Image                                                                                  
from facenet_pytorch import MTCNN                                                                      
import torch                                                                                           
                                                                                                    
device = torch.device("cpu")                                                                           
                                                                                                    
# Try with lower detection thresholds (more sensitive)                                                 
for path in ["selfie.jpg", "selfie3.jpg", "anchor_face.jpg"]:                                          
    img = Image.open(path).convert("RGB")                                                              
    print(f"\n{path}: size={img.size}")                                                                
                                                                                                    
    # Default MTCNN                                                                                    
    mtcnn_default = MTCNN(keep_all=True, device=device)                                                
    with torch.no_grad():                                                                              
        faces = mtcnn_default(img)                                                                     
    print(f"  Default thresholds: {'faces found' if faces is not None else 'NO FACES'}")               
                                                                                                    
    # More sensitive MTCNN (lower thresholds = more detections)                                        
    mtcnn_sensitive = MTCNN(                                                                           
        keep_all=True,                                                                                 
        device=device,                                                                                 
        thresholds=[0.5, 0.5, 0.5],  # Default is [0.6, 0.7, 0.7]                                      
        factor=0.709,  # Default                                                                       
        min_face_size=20,                                                                              
    )                                                                                                  
    with torch.no_grad():                                                                              
        faces = mtcnn_sensitive(img)                                                                   
    if faces is not None:                                                                              
        print(f"  Sensitive thresholds: {faces.shape[0]} face(s) found")                               
    else:                                                                                              
        print(f"  Sensitive thresholds: NO FACES")                                                     
                                                                                                    
    # Even more aggressive                                                                             
    mtcnn_aggressive = MTCNN(                                                                          
        keep_all=True,                                                                                 
        device=device,                                                                                 
        thresholds=[0.3, 0.4, 0.4],                                                                    
        factor=0.709,                                                                                  
        min_face_size=10,                                                                              
    )                                                                                                  
    with torch.no_grad():                                                                              
        faces = mtcnn_aggressive(img)                                                                  
    if faces is not None:                                                                              
        print(f"  Aggressive thresholds: {faces.shape[0]} face(s) found")                              
    else:                                                                                              
        print(f"  Aggressive thresholds: NO FACES")                