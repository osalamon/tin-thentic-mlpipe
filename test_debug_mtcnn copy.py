#!/usr/bin/env python3                                                                                 
"""Debug MTCNN face detection."""                                                                      
                                                                                                    
from PIL import Image                                                                                  
from facenet_pytorch import MTCNN                                                                      
import torch                                                                                           
                                                                                                    
device = torch.device("cpu")                                                                           
mtcnn = MTCNN(keep_all=True, device=device)                                                            
                                                                                                    
for path in ["selfie.jpg", "selfie3.jpg"]:                                                             
    img = Image.open(path).convert("RGB")                                                              
    print(f"\n{path}: size={img.size}, mode={img.mode}")                                               
                                                                                                    
    # Try detection at original size                                                                   
    with torch.no_grad():                                                                              
        faces = mtcnn(img)                                                                             
    print(f"  Original size: {'faces found' if faces is not None else 'NO FACES'}")                    
                                                                                                    
    # Try with a larger input (MTCNN likes faces at least ~50px)                                       
    w, h = img.size                                                                                    
    if min(w, h) < 400:                                                                                
        scale = 400 / min(w, h)                                                                        
        img_large = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)                        
        print(f"  Scaled up to {img_large.size}")                                                      
        with torch.no_grad():                                                                          
            faces = mtcnn(img_large)                                                                   
        print(f"  Scaled up: {'faces found' if faces is not None else 'NO FACES'}")   