"""
Module 2: Synthetic Sequence Generation Pipeline
=================================================
Implements the alpha-blending approach described in the project report (Section 4.3).

Key idea:
  For each (healthy_img, diseased_img) pair, produce 5 frames:
      frame_t = (1 - alpha_t) * healthy_img + alpha_t * diseased_img
  where alpha_values = [0.0, 0.25, 0.50, 0.75, 1.0]

This converts any static PlantVillage-style dataset into temporal sequences
suitable for training the CNN-LSTM Disease Progression Detection Module (DPDM).

Usage:
    python synthetic_sequence_generator.py

Author: AgroSmart Team
Report Section: 4.3 / RO5
"""

import os
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm
import json
import random

# ─────────────────────────────────────────────
# Configuration (mirrors report Table 4.3)
# ─────────────────────────────────────────────
TARGET_SIZE: Tuple[int, int] = (224, 224)   # H × W fed to MobileNetV2 / EfficientNetB3
ALPHA_VALUES: List[float] = [0.0, 0.25, 0.50, 0.75, 1.0]  # 5-frame progression
SEQUENCE_LENGTH: int = len(ALPHA_VALUES)    # 5
RANDOM_SEED: int = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ─────────────────────────────────────────────
# Image utilities
# ─────────────────────────────────────────────

def load_and_resize(path: str) -> np.ndarray:
    """
    Load an image from disk, convert to RGB, resize to TARGET_SIZE,
    and normalise pixel values to [0, 1] float32.

    Args:
        path: Absolute or relative path to the image file.

    Returns:
        numpy array of shape (H, W, 3), dtype float32, range [0, 1].
    """
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)
    return img.astype(np.float32) / 255.0


# ─────────────────────────────────────────────
# Core alpha-blending logic (Report Section 4.3.2)
# ─────────────────────────────────────────────

def generate_blended_sequence(
    healthy_img: np.ndarray,
    diseased_img: np.ndarray,
    alphas: List[float] = ALPHA_VALUES,
) -> np.ndarray:
    """
    Create an alpha-blended 5-frame sequence that simulates progressive
    disease onset from a fully healthy leaf to a fully diseased leaf.

    Formula:   frame_t = (1 - alpha) * healthy + alpha * diseased

    Args:
        healthy_img:  Float32 [0,1] array of shape (H, W, 3).
        diseased_img: Float32 [0,1] array of shape (H, W, 3).
        alphas:       Blending factors, default [0.0, 0.25, 0.50, 0.75, 1.0].

    Returns:
        numpy array of shape (T, H, W, 3) where T = len(alphas).
    """
    sequence = []
    for alpha in alphas:
        blended = (1.0 - alpha) * healthy_img + alpha * diseased_img
        blended = np.clip(blended, 0.0, 1.0)
        sequence.append(blended)
    return np.stack(sequence, axis=0)  # (T, H, W, 3)


# ─────────────────────────────────────────────
# Temporally-coherent augmentation
# ─────────────────────────────────────────────

def augment_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Apply consistent spatial augmentation across every frame in a sequence
    so that temporal coherence is preserved (same transform for all frames).

    Augmentations applied:
        • Random horizontal flip (p=0.5)
        • Random brightness jitter [0.8, 1.2]
        • Random rotation [-15°, +15°] with reflect border

    Args:
        sequence: Float32 array of shape (T, H, W, 3), range [0, 1].

    Returns:
        Augmented float32 array of same shape.
    """
    T, H, W, C = sequence.shape

    # ── Horizontal flip ──────────────────────────────────────
    if np.random.rand() > 0.5:
        sequence = sequence[:, :, ::-1, :].copy()

    # ── Brightness jitter (same factor for all frames) ───────
    brightness_factor = np.random.uniform(0.8, 1.2)
    sequence = np.clip(sequence * brightness_factor, 0.0, 1.0)

    # ── Rotation (same angle for all frames) ─────────────────
    angle = np.random.uniform(-15.0, 15.0)
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle, 1.0)
    rotated_frames = []
    for frame in sequence:
        frame_uint8 = (frame * 255).astype(np.uint8)
        rotated = cv2.warpAffine(
            frame_uint8, M, (W, H),
            borderMode=cv2.BORDER_REFLECT
        )
        rotated_frames.append(rotated.astype(np.float32) / 255.0)
    return np.stack(rotated_frames, axis=0)


# ─────────────────────────────────────────────
# Dataset builder (Report Listing 4.3)
# ─────────────────────────────────────────────

def build_dataset_from_plantvillage(
    dataset_root: str,
    output_dir: str,
    augment: bool = True,
    augment_copies: int = 2,
    max_pairs_per_class: int = 500,
) -> Dict[str, int]:
    """
    Build a synthetic temporal sequence dataset from the PlantVillage
    folder structure.

    Expected folder layout:
        dataset_root/
            <Crop>___<Disease>/     ← diseased class
            <Crop>___healthy/       ← healthy class for that crop
            ...

    For every (healthy, diseased) image pair:
        1. Alpha-blend into a 5-frame progression sequence.
        2. Optionally apply augmentation N times.
        3. Save each sequence as a .npy file under output_dir/<class_name>/.

    Args:
        dataset_root:        Path to PlantVillage root directory.
        output_dir:          Where to write generated .npy sequences.
        augment:             Whether to apply random augmentation.
        augment_copies:      Number of augmented copies per original pair.
        max_pairs_per_class: Cap on (healthy, diseased) pairs per class
                             to keep dataset balanced.

    Returns:
        Dictionary mapping class_name -> number of sequences generated.
    """
    dataset_root = Path(dataset_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Discover healthy and diseased class folders ───────────
    all_dirs = [d for d in dataset_root.iterdir() if d.is_dir()]

    # Group by crop: key = crop_name, value = {healthy: Path, diseases: [Path]}
    crop_groups: Dict[str, Dict] = {}
    for d in all_dirs:
        name = d.name
        if '___' in name:
            parts = name.split('___')
            crop = parts[0].lower()
            label = parts[1].lower()
        else:
            # e.g. "Tomato_healthy"
            crop = name.split('_')[0].lower()
            label = '_'.join(name.split('_')[1:]).lower()

        if crop not in crop_groups:
            crop_groups[crop] = {'healthy': None, 'diseases': []}

        if 'healthy' in label:
            crop_groups[crop]['healthy'] = d
        else:
            crop_groups[crop]['diseases'].append(d)

    stats: Dict[str, int] = {}
    seq_id_global = 0

    for crop, group in crop_groups.items():
        if group['healthy'] is None or not group['diseases']:
            print(f"[SKIP] {crop}: missing healthy or disease folder.")
            continue

        # Collect healthy image paths
        healthy_paths = _collect_images(group['healthy'])
        if not healthy_paths:
            print(f"[SKIP] {crop}: no healthy images found.")
            continue

        for disease_dir in group['diseases']:
            class_name = disease_dir.name
            disease_paths = _collect_images(disease_dir)
            if not disease_paths:
                continue

            class_out = output_dir / class_name
            class_out.mkdir(parents=True, exist_ok=True)

            # Sample pairs
            h_sample = random.sample(
                healthy_paths, min(max_pairs_per_class, len(healthy_paths))
            )
            d_sample = random.sample(
                disease_paths, min(max_pairs_per_class, len(disease_paths))
            )

            seq_id_class = 0
            pairs = list(zip(
                random.choices(h_sample, k=min(max_pairs_per_class, len(d_sample))),
                d_sample[:max_pairs_per_class]
            ))

            for h_path, d_path in tqdm(pairs, desc=f"Generating {class_name}"):
                try:
                    h_img = load_and_resize(str(h_path))
                    d_img = load_and_resize(str(d_path))
                except Exception as e:
                    print(f"  [WARN] Skipping pair: {e}")
                    continue

                # Original (no augmentation)
                seq = generate_blended_sequence(h_img, d_img)
                np.save(class_out / f"seq_{seq_id_global:07d}.npy", seq)
                seq_id_global += 1
                seq_id_class += 1

                # Augmented copies
                if augment:
                    for _ in range(augment_copies):
                        aug_seq = augment_sequence(seq.copy())
                        np.save(class_out / f"seq_{seq_id_global:07d}.npy", aug_seq)
                        seq_id_global += 1
                        seq_id_class += 1

            stats[class_name] = seq_id_class
            print(f"  ✓ {class_name}: {seq_id_class} sequences")

    # ── Save metadata ─────────────────────────────────────────
    meta = {
        "total_sequences": seq_id_global,
        "sequence_length": SEQUENCE_LENGTH,
        "alpha_values": ALPHA_VALUES,
        "image_size": list(TARGET_SIZE),
        "class_counts": stats,
        "class_names": sorted(stats.keys()),
    }
    with open(output_dir / "dataset_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n✅ Dataset generation complete.")
    print(f"   Total sequences : {seq_id_global}")
    print(f"   Classes         : {len(stats)}")
    print(f"   Saved to        : {output_dir}")
    return stats


def build_dataset_simple(
    healthy_dir: str,
    disease_classes: Dict[str, str],
    output_dir: str,
    augment: bool = True,
) -> None:
    """
    Simplified builder matching Report Listing 4.3 exactly.
    Useful for single-crop experiments.

    Args:
        healthy_dir:     Directory of healthy leaf images.
        disease_classes: Dict mapping class_name -> directory of diseased images.
        output_dir:      Root output directory for .npy sequences.
        augment:         Apply augmentation flag.
    """
    healthy_paths = _collect_images(Path(healthy_dir))
    os.makedirs(output_dir, exist_ok=True)

    seq_id = 0
    for class_name, disease_dir in disease_classes.items():
        disease_paths = _collect_images(Path(disease_dir))
        class_output = os.path.join(output_dir, class_name)
        os.makedirs(class_output, exist_ok=True)

        for h_path in tqdm(healthy_paths, desc=f"Generating {class_name}"):
            healthy_img = load_and_resize(str(h_path))
            for d_path in disease_paths:
                diseased_img = load_and_resize(str(d_path))
                sequence = generate_blended_sequence(healthy_img, diseased_img)
                if augment:
                    sequence = augment_sequence(sequence)
                out_path = os.path.join(class_output, f"seq_{seq_id:06d}.npy")
                np.save(out_path, sequence)
                seq_id += 1

    print(f"Dataset generation complete. Total sequences: {seq_id}")


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _collect_images(directory: Path) -> List[Path]:
    """Return all JPEG/PNG image paths inside a directory."""
    exts = ["*.jpg", "*.JPG", "*.jpeg", "*.JPEG", "*.png", "*.PNG"]
    paths = []
    for ext in exts:
        paths.extend(directory.glob(ext))
    return paths


def verify_sequence(npy_path: str) -> bool:
    """
    Quick sanity check for a saved sequence file.

    Args:
        npy_path: Path to a .npy sequence file.

    Returns:
        True if valid, False otherwise.
    """
    try:
        seq = np.load(npy_path)
        assert seq.shape == (SEQUENCE_LENGTH, TARGET_SIZE[0], TARGET_SIZE[1], 3), \
            f"Unexpected shape: {seq.shape}"
        assert seq.dtype == np.float32, f"Unexpected dtype: {seq.dtype}"
        assert 0.0 <= seq.min() and seq.max() <= 1.0, "Pixel values out of [0,1]"
        return True
    except Exception as e:
        print(f"[ERROR] Invalid sequence {npy_path}: {e}")
        return False


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    # Paths — adjust to your local setup
    PLANTVILLAGE_ROOT = "../../datasets/plant_disease/PlantVillage"
    OUTPUT_DIR = "../../datasets/synthetic_sequences"

    print("=" * 60)
    print("  SYNTHETIC SEQUENCE GENERATION PIPELINE")
    print("  Method: Alpha Blending (Report Section 4.3)")
    print("=" * 60)

    stats = build_dataset_from_plantvillage(
        dataset_root=PLANTVILLAGE_ROOT,
        output_dir=OUTPUT_DIR,
        augment=True,
        augment_copies=2,
        max_pairs_per_class=300,
    )

    # Verify a random sample of generated sequences
    print("\nVerifying random samples...")
    sample_classes = list(stats.keys())[:3]
    for cls in sample_classes:
        cls_dir = Path(OUTPUT_DIR) / cls
        seqs = list(cls_dir.glob("*.npy"))
        if seqs:
            ok = verify_sequence(str(seqs[0]))
            print(f"  {cls}: {'✓ OK' if ok else '✗ FAILED'}")
