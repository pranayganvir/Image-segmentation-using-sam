import os
import torch
import torchvision
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import cv2
import supervision as sv


#Defining Model tyoe and device
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model_type = "vit_h"
sam_checkpoint = r"C:\Users\LOQ\Desktop\image_segmentation\sam_vit_h_4b8939.pth"
device = "cuda"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
mask_generator = SamAutomaticMaskGenerator(sam)


def segmented_pipeline(filename):
    
    #Step-01: Read Image
    img = cv2.imread(filename) #BGR
    
    #Step-02: convert into RGB
    image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #Step-03: Applying Mask Generator to the image
    sam_result = mask_generator.generate(image_rgb)
    
    #Step-04: Defining mask annoatator
    mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)
    
    #Step-05: Detecting the object
    detections = sv.Detections.from_sam(sam_result=sam_result)
    
    #Step-06: applying mask_annotator
    annotated_image = mask_annotator.annotate(scene=img.copy(), detections=detections)
    
    return annotated_image
    
    
    