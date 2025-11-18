#!/usr/bin/env python
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from torchvision import transforms
from PIL import Image

import timm
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine
from difflib import SequenceMatcher

import optuna


# =============================================================================
# Utility functions
# =============================================================================

def accuracy(target, pred):
    return accuracy_score(
        target.detach().cpu().numpy(),
        pred.detach().cpu().numpy()
    )


def euclidean_similarity(v1, v2):
    """Calculate similarity using Euclidean distance."""
    v1, v2 = np.asarray(v1).flatten(), np.asarray(v2).flatten()
    return 1.0 / (1.0 + np.linalg.norm(v1 - v2))


def cosine_sim(v1, v2):
    """Calculate cosine similarity."""
    v1, v2 = np.asarray(v1).flatten(), np.asarray(v2).flatten()
    return 1.0 - cosine(v1, v2)


def jaccard_similarity(set1, set2):
    """Calculate Jaccard similarity between two sets or arrays."""
    a, b = np.asarray(set1).ravel(), np.asarray(set2).ravel()
    a, b = set(a.tolist()), set(b.tolist())
    intersection = len(a & b)
    union = len(a | b)
    return intersection / union if union != 0 else 0.0


def pearson_correlation(v1, v2):
    """Calculate Pearson correlation coefficient."""
    v1, v2 = np.asarray(v1).flatten(), np.asarray(v2).flatten()
    return np.corrcoef(v1, v2)[0, 1]


def string_similarity(str1, str2):
    """Calculate string similarity using SequenceMatcher."""
    str1, str2 = str(str1), str(str2)
    return SequenceMatcher(None, str1, str2).ratio()


def compare_similarities_vecs(test_vec, gt_vecs, sim_func):
    """Compute similarity between one test vector and many GT vectors."""
    tv = np.asarray(test_vec).ravel()
    sims = []
    for g in gt_vecs:
        gv = np.asarray(g).ravel()
        if tv.shape[0] != gv.shape[0]:
            raise ValueError(
                f"Dim mismatch: test D={tv.shape[0]} vs GT D={gv.shape[0]}"
            )
        sims.append(sim_func(tv, gv))
    return np.array(sims)


# =============================================================================
# Dataset
# =============================================================================

class MVTecDataset(Dataset):
    def __init__(self, root_dir, category, split="train", transform=None):
        """
        Args:
            root_dir: Path to MVTec AD dataset root
            category: Product category (e.g., 'bottle', 'cable', 'capsule')
            split: 'train' or 'test'
            transform: Preprocessing transforms
        """
        self.root_dir = Path(root_dir)
        self.category = category
        self.split = split
        self.transform = transform

        self.image_paths: List[Path] = []
        self.labels: List[int] = []
        self.defect_types: List[str] = []

        if split == "train":
            # Training images are all normal
            train_dir = self.root_dir / category / "train" / "good"
            for img_path in sorted(train_dir.glob("*.png")):
                self.image_paths.append(img_path)
                self.labels.append(0)
                self.defect_types.append("good")
        else:
            # Test images include normal and anomalies
            test_dir = self.root_dir / category / "test"
            for defect_dir in sorted(test_dir.iterdir()):
                if defect_dir.is_dir():
                    defect_type = defect_dir.name
                    label = 0 if defect_type == "good" else 1
                    for img_path in sorted(defect_dir.glob("*.png")):
                        self.image_paths.append(img_path)
                        self.labels.append(label)
                        self.defect_types.append(defect_type)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        # Note: we return the defect_type text label, same as your notebook
        return image, self.defect_types[idx]


# =============================================================================
# Embedding with DINO (timm)
# =============================================================================

def get_transforms(image_size=224,
                   mean=(0.485, 0.456, 0.406),
                   std=(0.229, 0.224, 0.225)):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def embed_and_save_features(
    dataloader: DataLoader,
    model_name: str = "vit_small_patch16_dinov3_qkvb.lvd1689m",
    device: str = "cpu",
) -> List[Tuple[np.ndarray, str]]:
    """
    For one batch from the dataloader, compute DINO embeddings & return
    list of (embedding, label_str).
    """
    # Load model
    model = timm.create_model(model_name, pretrained=True)
    model.eval()
    model.to(device)

    # Preprocessing (timm's eval pipeline) + tensor->PIL converter
    data_config = timm.data.resolve_model_data_config(model)
    preprocess = timm.data.create_transform(**data_config, is_training=False)
    to_pil = transforms.ToPILImage()

    images, labels = next(iter(dataloader))

    embeddings = []
    for img, label in zip(images, labels):
        # Ensure PIL for timm transforms
        if isinstance(img, torch.Tensor):
            img = to_pil(img)

        x = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            feats = model.forward_features(x)
            if isinstance(feats, dict):
                emb = feats.get("x_norm_clstoken") or feats.get("x_norm")
            else:
                emb = feats

        embeddings.append((emb.squeeze(0).cpu().numpy(), label))

    return embeddings


# =============================================================================
# Anomaly decision + accuracy
# =============================================================================

def compare_and_draw_conclusion(
    embeddings_GT,
    embeddings_Test,
    threshold: float = 0.8,
    percentile: float = 0.1,
) -> float:
    """
    - percentile: float between 0 and 1 indicating which percentile similarity to use.
    - Uses the percentile-th best similarity.
    - Predicts 'good' if that similarity > threshold, else 'anomalous'.
    - Returns accuracy over all test embeddings.
    """
    if not embeddings_GT or not embeddings_Test:
        return 0.0

    gt_embeddings, _ = zip(*embeddings_GT)

    correct = 0
    total = 0

    for test_embedding, label in embeddings_Test:
        similarities = compare_similarities_vecs(
            test_embedding, gt_embeddings, cosine_sim
        )
        sorted_sims = np.sort(similarities)[::-1]  # descending order

        rank = max(1, int(percentile * len(sorted_sims)))
        rank = min(rank, len(sorted_sims))
        selected_score = sorted_sims[rank - 1]

        predicted_label = "good" if selected_score > threshold else "anomalous"
        true_label = "good" if str(label).lower() == "good" else "anomalous"

        is_correct = int(predicted_label == true_label)
        correct += is_correct
        total += 1

    accuracy = correct / total if total else 0.0
    return accuracy


# =============================================================================
# Optuna study
# =============================================================================

def run_study(
    embeddings_GT,
    embeddings_Test,
    percentile_min: float,
    percentile_max: float,
    threshold_min: float,
    threshold_max: float,
    n_trials: int,
    study_name: str = None,
    storage: str = None,
    load_if_exists: bool = False,
) -> optuna.study.Study:

    def objective(trial: optuna.trial.Trial) -> float:
        percentile = trial.suggest_float(
            "percentile", percentile_min, percentile_max
        )
        threshold = trial.suggest_float(
            "threshold", threshold_min, threshold_max
        )

        score = compare_and_draw_conclusion(
            embeddings_GT=embeddings_GT,
            embeddings_Test=embeddings_Test,
            percentile=percentile,
            threshold=threshold,
        )
        return score

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=storage,
        load_if_exists=load_if_exists,
    )
    study.optimize(objective, n_trials=n_trials)

    return study


# =============================================================================
# Main / CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Optuna study for DINO-based anomaly detection on MVTec."
    )

    # Data / model
    parser.add_argument(
        "--mvtec-root",
        type=str,
        default="./Dataset",
        help="Root directory of MVTec AD dataset.",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="bottle,cable,capsule,carpet,grid,hazelnut,leather,metal_nut,pill,screw,tile,toothbrush,transistor,wood,zipper",
        help="Comma-separated list of categories to include.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for dataloaders (only the first batch is used for embeddings).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Image size for preliminary transforms.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="vit_small_patch16_dinov3_qkvb.lvd1689m",
        help="timm model name for DINO embeddings.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on (e.g. 'cpu' or 'cuda').",
    )

    # Optuna search space
    parser.add_argument(
        "--percentile-min",
        type=float,
        default=0.0,
        help="Lower bound for percentile hyperparameter.",
    )
    parser.add_argument(
        "--percentile-max",
        type=float,
        default=1.0,
        help="Upper bound for percentile hyperparameter.",
    )
    parser.add_argument(
        "--threshold-min",
        type=float,
        default=0.4,
        help="Lower bound for threshold hyperparameter.",
    )
    parser.add_argument(
        "--threshold-max",
        type=float,
        default=1.0,
        help="Upper bound for threshold hyperparameter.",
    )

    # Optuna study config
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials.",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name (optional).",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., 'sqlite:///study.db'). If None, uses in-memory.",
    )
    parser.add_argument(
        "--load-if-exists",
        action="store_true",
        help="If storage is set and study with same name exists, load it.",
    )

    return parser.parse_args()


def build_dataloaders(
    root: str,
    categories: List[str],
    batch_size: int,
    image_size: int,
):
    transform = get_transforms(image_size=image_size)

    train_datasets = [
        MVTecDataset(root_dir=root, category=cat, split="train", transform=transform)
        for cat in categories
    ]
    test_datasets = [
        MVTecDataset(root_dir=root, category=cat, split="test", transform=transform)
        for cat in categories
    ]

    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )

    return train_loader, test_loader


def main():
    args = parse_args()

    categories = [c.strip() for c in args.categories.split(",") if c.strip()]

    print("=== Settings ===")
    print(f"MVTec root    : {args.mvtec_root}")
    print(f"Categories    : {categories}")
    print(f"Batch size    : {args.batch_size}")
    print(f"Image size    : {args.image_size}")
    print(f"Model name    : {args.model_name}")
    print(f"Device        : {args.device}")
    print(f"n_trials      : {args.n_trials}")
    print(f"percentile in : [{args.percentile_min}, {args.percentile_max}]")
    print(f"threshold in  : [{args.threshold_min}, {args.threshold_max}]")
    print("================")

    # Build dataloaders
    train_loader, test_loader = build_dataloaders(
        root=args.mvtec_root,
        categories=categories,
        batch_size=args.batch_size,
        image_size=args.image_size,
    )

    # Precompute embeddings (once!)
    print("Computing training embeddings (GT)...")
    embeddings_GT = embed_and_save_features(
        train_loader,
        model_name=args.model_name,
        device=args.device,
    )

    print("Computing test embeddings...")
    embeddings_Test = embed_and_save_features(
        test_loader,
        model_name=args.model_name,
        device=args.device,
    )

    # Run Optuna study
    print("Running Optuna study...")
    study = run_study(
        embeddings_GT=embeddings_GT,
        embeddings_Test=embeddings_Test,
        percentile_min=args.percentile_min,
        percentile_max=args.percentile_max,
        threshold_min=args.threshold_min,
        threshold_max=args.threshold_max,
        n_trials=args.n_trials,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=args.load_if_exists,
    )

    print("\n=== Study complete ===")
    print(f"Best value      : {study.best_value:.6f}")
    print("Best parameters :")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
