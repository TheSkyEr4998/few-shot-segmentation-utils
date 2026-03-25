IMG_SIZE = 128

def load_image_mask(img_path, mask_path):
    img = Image.open(img_path).convert("L").resize((IMG_SIZE, IMG_SIZE))
    mask = Image.open(mask_path).convert("L").resize((IMG_SIZE, IMG_SIZE))

    img = np.array(img) / 255.0
    mask = (np.array(mask) > 127).astype(np.float32)

    return img[..., np.newaxis], mask[..., np.newaxis]


def sample_task(task_path, k_shot=3, q_query=2):
    img_dir = os.path.join(task_path, "images")
    mask_dir = os.path.join(task_path, "masks")

    files = sorted(os.listdir(img_dir))
    idx = list(range(len(files)))
    random.shuffle(idx)

    support_idx = idx[:k_shot]
    query_idx = idx[k_shot:k_shot+q_query]

    support, query = [], []

    for i in support_idx:
        support.append(load_image_mask(
            os.path.join(img_dir, files[i]),
            os.path.join(mask_dir, files[i])
        ))

    for i in query_idx:
        query.append(load_image_mask(
            os.path.join(img_dir, files[i]),
            os.path.join(mask_dir, files[i])
        ))

    return support, query
