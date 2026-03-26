import os
import random
import cv2
import numpy as np

def get_task_data(base_path, split, task_id):
    task_path = os.path.join(base_path, split, task_id)
    img_dir = os.path.join(task_path, 'images')
    mask_dir = os.path.join(task_path, 'masks')
    
    images, masks = [], []
    filenames = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    
    for f in filenames:
        img = cv2.imread(os.path.join(img_dir, f), cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(os.path.join(mask_dir, f), cv2.IMREAD_GRAYSCALE)
        # Normalization to [0, 1]
        images.append(np.expand_dims(img / 255.0, -1))
        masks.append(np.expand_dims(mask / 255.0, -1))
        
    return np.array(images), np.array(masks)

def task_generator(base_path, split='train', batch_size=4):
    """Generator for task-based sampling."""
    split_path = os.path.join(base_path, split)
    tasks = [t for t in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, t))]
    
    while True:
        task = random.choice(tasks)
        images, masks = get_task_data(base_path, split, task)
        
        if len(images) < batch_size:
            idx = np.arange(len(images))
        else:
            idx = np.random.choice(len(images), batch_size, replace=False)
            
        yield images[idx], masks[idx]
