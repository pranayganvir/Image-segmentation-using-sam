
# Image Segmentation Using Facebook's Segment Anything Model (SAM)


This project leverages Facebook's **Segment Anything Model (SAM)** for performing advanced image segmentation. SAM is a state-of-the-art model capable of segmenting any object in an image with minimal effort. The application is built to demonstrate the power of SAM in real-world scenarios, offering an intuitive web interface to upload images and visualize segmentation results.

## Image Segmentation
Image segmentation is a computer vision technique that assigns a label to every pixel in an image such that pixels with the same label share certain characteristics.

For example, in a street scene, all pixels belonging to cars might be labeled with one color, while those belonging to the road might be labeled with another.

## 

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

The Segment Anything Model (SAM) produces high-quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1 billion masks and has strong zero-shot performance on a variety of segmentation tasks.

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Features

- **State-of-the-Art Segmentation**:
  - Utilizes Facebook's **Segment Anything Model** to accurately identify and segment objects in images.
  - Supports both automatic and manual segmentation modes.

- **Web Interface**:
  - Upload images directly via an intuitive web application.
  - Visualize segmented outputs in real-time.

- **Customizable**:
  - Adjust parameters for segmentation.
  - Option to save segmented outputs locally.

- **Deployment-Ready**:
  - Easy to deploy using Flask or any other web framework.
  - Scalable and suitable for integration into larger pipelines.

---

## **Getting Started** 🚀

### Now let’s dive into its working and implementations.

- If you want to use Segment Anything on a local machine, make sure to Create a new environment and install some required libraries such as python>=3.8, pytorch>=1.7, and torchvision>=0.8, Or you can use the Google Colab to run the code. Here we will be using a Colab notebook for the example.

- In the Colab notebook, ensure you have access to a GPU for faster processing.

- Clone the repository and install.

--- 

1. Clone the repository:
   ```bash
   !pip install -q 'git+https://github.com/facebookresearch/segment-anything.git'
   !pip install -q jupyter_bbox_widget roboflow dataclasses-json supervision==0.23.0
   ```

2. Import required libraries.
``` bash
import torch
import torchvision
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import supervision as sv
import zipfile
```
3. Download SAM weights:
```bash
!mkdir -p {HOME}/weights
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -P {HOME}/weights
CHECKPOINT_PATH = os.path.join(HOME, "weights", "sam_vit_h_4b8939.pth")
print(CHECKPOINT_PATH, "; exist:", os.path.isfile(CHECKPOINT_PATH)
```

4. Load Model
```bash
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT_PATH).to(device=DEVICE)
mask_generator = SamAutomaticMaskGenerator(sam)
```

5. Generate masks with SAM:
```bash
image_bgr = cv2.imread("/content/pranav-nahata-6ttYvAR7yio-unsplash.jpg")
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
sam_result = mask_generator.generate(image_rgb)
```

5. Results visualisation with Supervision:
```bash
mask_annotator = sv.MaskAnnotator(color_lookup=sv.ColorLookup.INDEX)

detections = sv.Detections.from_sam(sam_result=sam_result)

annotated_image = mask_annotator.annotate(scene=image_bgr.copy(), detections=detections)

sv.plot_images_grid(
    images=[image_bgr, annotated_image],
    grid_size=(1, 2),
    titles=['source image', 'segmented image']
)
```

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)






