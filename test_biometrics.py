#!/usr/bin/env python3                                                                                 
"""Quick validation script for BiometricValidator."""                                                  
                                                                                                    
import logging                                                                                         
from pathlib import Path                                                                               
                                                                                                    
import torch                                                                                           
from PIL import Image                                                                                  
from facenet_pytorch import MTCNN, InceptionResnetV1                                                   
                                                                                                    
from src.config import IngestConfig                                                                    
                                                                                                    
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")                           
                                                                                                    
config = IngestConfig(ANCHOR_FACE_PATH=Path("anchor_face.jpg"))                                        
                                                                                                    
# Load models manually for debugging                                                                   
device = torch.device("cpu")                                                                           
mtcnn = MTCNN(keep_all=True, device=device)                                                            
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)                                    
                                                                                                    
def get_embedding(image_path):                                                                         
    img = Image.open(image_path).convert("RGB")                                                        
    with torch.no_grad():                                                                              
        faces = mtcnn(img)                                                                             
        if faces is None:                                                                              
            return None                                                                                
        if faces.ndim == 3:                                                                            
            faces = faces.unsqueeze(0)                                                                 
        embeddings = resnet(faces.to(device))                                                          
    return embeddings                                                                                  
                                                                                                    
# Get anchor embedding                                                                                 
anchor_emb = get_embedding("anchor_face.jpg")                                                          
if anchor_emb is None:                                                                                 
    print("ERROR: No face found in anchor_face.jpg!")                                                  
    exit(1)                                                                                            
anchor_emb = anchor_emb[0]  # Use first face                                                           
                                                                                                    
# Test each image                                                                                      
test_images = [                                                                                        
    ("selfie.jpg", True),                                                                              
    ("selfie2.jpg", True),                                                                             
    ("selfie3.jpg", True),                                                                             
    ("landscape.jpg", False),                                                                          
    ("stranger.jpg", False),                                                                           
]                                                                                                      
                                                                                                    
print("\n--- Distance Analysis ---\n")                                                                 
for path, expected in test_images:                                                                     
    embs = get_embedding(path)                                                                         
    if embs is None:                                                                                   
        print(f"{path}: NO FACES DETECTED")                                                            
        continue                                                                                       
                                                                                                    
    for i, emb in enumerate(embs):                                                                     
        dist = torch.nn.functional.pairwise_distance(                                                  
            emb.unsqueeze(0), anchor_emb.unsqueeze(0)                                                  
        ).item()                                                                                       
        match = "MATCH" if dist <= config.FACE_DISTANCE_TOLERANCE else "NO MATCH"                      
        print(f"{path} face#{i}: distance={dist:.4f} ({match})")                                       
    print() 