import os
import torch
import timm
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, ConcatDataset
import faiss
from tqdm import tqdm
import csv
from joblib import dump, load
import torch.nn.functional as F

# ====================== CONFIG ======================
MVTEC_ROOT = "../Dataset"
BATCH_SIZE = 32
IMAGE_SIZE = 224
NUM_WORKERS = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMBEDDING_CACHE = "./embeddings_patch"
os.makedirs(EMBEDDING_CACHE, exist_ok=True)

# DINOv3 model with patch output
MODEL_NAME = "vit_small_patch16_dinov3_qkvb.lvd1689m"  # or _dist
PATCH_SIZE = 16
FEAT_DIM = 384  # ViT-S/16 DINOv3

# Anomaly detection params
K = 9                    # Use 9-NN (standard in PatchCore)
CORESET_FRAC = 0.1       # Keep only 10% of patches → huge memory saving, little accuracy drop
DOWNSAMPLE_P = 4         # Optional: spatial downsampling (e.g. keep every 4th patch)

# ===================================================

from torchvision import transforms
from torchvision.datasets import ImageFolder  # or use your MVTecDataset

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, split="train", transform=None, cache=False):
        """
        Args:
            root_dir: Path to MVTec AD dataset root.
            category: Product category (e.g., 'bottle', 'cable', 'capsule').
            split: 'train' or 'test'.
            transform: torchvision transforms.
            cache: If True, images are pre-loaded into RAM for faster epochs.
        """
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.transform = transform
        self.cache = cache

        base_dir = self.root_dir / category / split

        if split == "train":
            img_dirs = [(base_dir / "good", 0, "good")]
        else:
            # Each folder inside 'test' is a defect type
            img_dirs = [
                (d, 0 if d.name == "good" else 1, d.name)
                for d in base_dir.iterdir()
                if d.is_dir()
            ]

        self.image_paths = []
        self.labels = []
        self.defect_types = []

        for folder, lbl, defect in img_dirs:
            for img_path in sorted(folder.glob("*.png")):
                self.image_paths.append(img_path)
                self.labels.append(lbl)
                self.defect_types.append(defect)

        # Optional RAM caching
        self.cached_images = None
        if cache:
            self.cached_images = [
                Image.open(p).convert("RGB") for p in self.image_paths
            ]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # if self.cache:
        #     img = self.cached_images[idx]
        # else:
        img = Image.open(self.image_paths[idx]).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = self.labels[idx]
        defect_type = self.defect_types[idx]

        return img, defect_type
# torch.multiprocessing.set_sharing_strategy("file_system")
# torch.multiprocessing.set_start_method("spawn", force=True)

MVTEC_ROOT = "../Dataset"
BATCH_SIZE = 40
IMAGE_SIZE = 224
NUM_WORKERS = min(8, os.cpu_count() - 1)

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]

root_dir = Path(MVTEC_ROOT)
all_categories = [cat.name for cat in root_dir.iterdir() if cat.is_dir()] 
def build_concat(split):
    datasets = [
        MVTecDataset(MVTEC_ROOT, cat, split=split, transform=transform)
        for cat in all_categories
    ]
    return ConcatDataset(datasets)


# Your existing dataset setup (assumed correct)
train_dataset = build_concat("train")
test_dataset  = build_concat("test")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True)


# ====================== MODEL ======================
@torch.no_grad()
def get_dinov3_patch_model():
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)
    model = model.to(DEVICE)
    model.eval()

    # Hook to get patch tokens (before final norm & head)
    feat_out = {}
    def hook(module, input, output):
        feat_out["feats"] = output  # (B, N+1, D): includes CLS + patches
    model.norm.register_forward_hook(hook)  # or blocks[-1].norm1 if needed
    return model, feat_out

model, feat_out = get_dinov3_patch_model()


# ====================== PATCH EXTRACTION ======================
@torch.no_grad()
def extract_patch_features(loader, split_name):
    cache_path = f"{EMBEDDING_CACHE}/{split_name}_patches.joblib"
    if os.path.exists(cache_path):
        print(f"Loading cached {split_name} patches...")
        return load(cache_path)

    all_patches = []
    all_labels = []
    all_paths = []  # optional: for debugging

    print(f"Extracting patch features: {split_name}")
    for images, labels in tqdm(loader, desc=split_name):
        images = images.to(DEVICE)

        _ = model(images)  # forward pass
        feats = feat_out["feats"]  # (B, 1 + N_patches, D)

        # Remove CLS token, keep patches only
        patch_feats = feats[:, 1:, :]  # (B, 196, 384)

        # Optional: spatial downsampling (e.g. 4x4 grid instead of 14x14)
        if DOWNSAMPLE_P > 1:
            B, N, D = patch_feats.shape
            H = W = int(N**0.5)
            patch_feats = patch_feats.view(B, H, W, D)
            patch_feats = patch_feats[:, ::DOWNSAMPLE_P, ::DOWNSAMPLE_P, :].reshape(B, -1, D)

        patch_feats = patch_feats.cpu().numpy()

        for i in range(images.size(0)):
            all_patches.append(patch_feats[i])      # (N_p, D)
            all_labels.append(labels[i].item() if torch.is_tensor(labels[i]) else labels[i])

    # Stack all patches: (Total_Patches, D)
    patches_np = np.concatenate(all_patches, axis=0)
    print(f"{split_name} patches: {patches_np.shape}")

    dump((patches_np, all_labels), cache_path)
    print(f"Saved to {cache_path}")
    return patches_np, all_labels


# ====================== CORESET SELECTION (PatchCore-style) ======================
def build_coreset(patches, frac=CORESET_FRAC):
    print(f"Building coreset: {len(patches)} → ~{int(len(patches) * frac):,}")
    # Simple greedy coreset (approximation of PatchCore's method)
    index = faiss.IndexFlatL2(FEAT_DIM)
    index.add(patches.astype(np.float32))
    
    # Sample subset greedily
    sampled_idx = []
    for _ in tqdm(range(int(len(patches) * frac)), desc="Coreset"):
        if not sampled_idx:
            idx = np.random.randint(0, len(patches))
        else:
            dists = index.search(patches.astype(np.float32), k=1)[0].flatten()
            idx = np.argmax(dists)
        sampled_idx.append(idx)
        index.remove_ids(np.array([idx]))  # remove to avoid re-pick
        index.add(patches[idx:idx+1].astype(np.float32))

    return patches[sampled_idx]


# ====================== MAIN EVALUATION ======================
def run_patchcore_anomaly_detection():
    # Extract train (reference) patches
    train_patches, _ = extract_patch_features(train_loader, "train")

    # Build memory-efficient coreset
    train_coreset = build_coreset(train_patches, frac=CORESET_FRAC)
    train_coreset = train_coreset / np.linalg.norm(train_coreset, axis=1, keepdims=True)  # L2 normalize

    # Build FAISS index (GPU if available)
    dim = train_coreset.shape[1]
    res = faiss.StandardGpuResources() if DEVICE == "cuda" else None
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim for normalized vectors
    if res:
        index = faiss.index_cpu_to_gpu(res, 0, index)
    
    index.add(train_coreset.astype(np.float32))

    # Extract test patches
    test_patches_all, test_labels = extract_patch_features(test_loader, "test")

    results = []
    correct = 0
    total = 0

    print("Running inference on test set...")
    for i in tqdm(range(0, len(test_patches_all), 196//(DOWNSAMPLE_P**2) if DOWNSAMPLE_P > 1 else 196)):
        end_idx = i + (196 // (DOWNSAMPLE_P**2) if DOWNSAMPLE_P > 1 else 196)
        if end_idx > len(test_patches_all):
            break
        test_patches = test_patches_all[i:end_idx]  # one image
        true_label = "good" if test_labels[i] == 0 else "anomalous"

        test_patches = test_patches / np.linalg.norm(test_patches, axis=1, keepdims=True)
        test_patches = test_patches.astype(np.float32)

        # k-NN search
        scores, _ = index.search(test_patches, k=K)
        anomaly_score = -scores[:, -1].mean()  # negative of k-th nearest neighbor similarity

        pred_label = "anomalous" if anomaly_score > 0.25 else "good"  # tune this threshold!
        is_correct = pred_label == true_label

        correct += is_correct
        total += 1

        results.append({
            "Image": total,
            "True": true_label,
            "Pred": pred_label,
            "Score": f"{anomaly_score:.4f}",
            "Correct": is_correct
        })

    acc = correct / total
    print(f"Final Accuracy: {acc*100:.2f}%")

    with open("patchcore_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    return acc

# ====================== RUN ======================
if __name__ == "__main__":
    run_patchcore_anomaly_detection()
