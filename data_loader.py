import os
import random
import numpy as np
from PIL import Image

# ================================
# CONFIG
# ================================
IMG_SIZE = 128

# reproducibility (important for report consistency)
random.seed(42)
np.random.seed(42)


# ================================
# LOAD SINGLE IMAGE + MASK
# ================================
def load_image_mask(img_path, mask_path):
    """
    Loads and preprocesses a single image-mask pair.

    - Converts to grayscale
    - Resizes to IMG_SIZE x IMG_SIZE
    - Normalizes image to [0,1]
    - Converts mask to binary {0,1}
    """

    # load
    img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    mask = Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE))

    # convert to numpy
    img = np.array(img, dtype=np.float32) / 255.0
    mask = (np.array(mask) > 127).astype(np.float32)

    # add channel dimension
    img = img[..., np.newaxis]
    mask = mask[..., np.newaxis]

    return img, mask


# ================================
# SAMPLE FEW-SHOT TASK
# ================================
def sample_task(task_path, k_shot=3, q_query=2):
    """
    Samples a few-shot task:
    - k_shot → support set
    - q_query → query set

    Returns:
        support: list of (image, mask)
        query: list of (image, mask)
    """

    img_dir = os.path.join(task_path, "images")
    mask_dir = os.path.join(task_path, "masks")

    # get only valid image files
    files = sorted([
        f for f in os.listdir(img_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ])

    # ================================
    # SAFETY CHECKS
    # ================================
    assert len(files) > (k_shot + q_query), \
        f"Not enough samples in task: {task_path}"

    # ensure corresponding masks exist
    for f in files:
        mask_path = os.path.join(mask_dir, f)
        assert os.path.exists(mask_path), \
            f"Missing mask for {f} in {task_path}"

    # ================================
    # RANDOM SAMPLING
    # ================================
    indices = list(range(len(files)))
    random.shuffle(indices)

    support_idx = indices[:k_shot]
    query_idx = indices[k_shot:k_shot + q_query]

    support = []
    query = []

    # build support set
    for i in support_idx:
        img_path = os.path.join(img_dir, files[i])
        mask_path = os.path.join(mask_dir, files[i])
        support.append(load_image_mask(img_path, mask_path))

    # build query set
    for i in query_idx:
        img_path = os.path.join(img_dir, files[i])
        mask_path = os.path.join(mask_dir, files[i])
        query.append(load_image_mask(img_path, mask_path))

    return support, query
