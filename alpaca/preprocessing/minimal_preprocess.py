"""
Minimal preprocessing for ALPaCA when images are already:
- N4 Bias Corrected
- Co-registered
- Skull-stripped  
- Have labeled lesion candidates

Skips: registration, brain extraction, MIMoSA, lesion labeling
"""

import numpy as np
import nibabel as nib
from scipy.ndimage import binary_erosion
from pathlib import Path

def normalize_image(data: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Z-score normalize using precomputed mask."""
    
    mean = data[mask].mean()
    std = data[mask].std()
    
    normalized = np.zeros_like(data, dtype=np.float32)
    normalized[mask] = (data[mask] - mean) / std
    
    return normalized

def erode_labels(labels: np.ndarray, iterations: int = 1) -> np.ndarray:
    """Erode each lesion label separately."""

    eroded = np.zeros_like(labels)
    
    for label_id in np.unique(labels):
        if label_id == 0:
            continue
        
        mask = (labels == label_id)
        eroded_mask = binary_erosion(mask, iterations=iterations)
        eroded[eroded_mask] = label_id
    
    return eroded


def minimal_preprocess(
    t1_path: str, 
    flair_path: str, 
    epi_path: str, 
    phase_path: str,
    labels_path: str, 
    output_dir: str, 
    eroded_candidates_path: str = None,
    verbose: bool = True
):
    """Minimal preprocessing: normalize images and erode lesion labels."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("\n┌" + "─"*38 + "┐")
        print("│" + " Image Preprocessing ".center(38) + "│")
        print("└" + "─"*38 + "┘")   
    
    # [1/4] Load all images
    if verbose:
        print("[1/4] Loading images...")
    
    t1_nii = nib.load(t1_path)
    flair_nii = nib.load(flair_path)
    epi_nii = nib.load(epi_path)
    phase_nii = nib.load(phase_path)
    labels_nii = nib.load(labels_path)
    
    t1 = t1_nii.get_fdata(dtype=np.float32)
    flair = flair_nii.get_fdata(dtype=np.float32)
    epi = epi_nii.get_fdata(dtype=np.float32)
    phase = phase_nii.get_fdata(dtype=np.float32)
    labels = labels_nii.get_fdata().astype(np.int32)
    
    # [2/4] Normalize images
    if verbose:
        print("[2/4] Normalizing modalities...")
    
    t1_norm = normalize_image(t1, t1 > 0)
    flair_norm = normalize_image(flair, flair > 0)
    epi_norm = normalize_image(epi, epi > 0)
    phase_norm = normalize_image(phase, phase > 0)
    
    # [3/4] Erode labels
    if verbose:
        print("[3/4] Eroding lesion candidates...")
    
    if eroded_candidates_path is not None:
        eroded = nib.load(eroded_candidates_path).get_fdata().astype(np.int16)
    else:
        eroded = erode_labels(labels, iterations=1)
    
    # [4/4] Save preprocessed files
    if verbose:
        print("[4/4] Saving preprocessed files...")
    
    t1_out = output_dir / "t1_final.nii.gz"
    flair_out = output_dir / "flair_final.nii.gz"
    epi_out = output_dir / "epi_final.nii.gz"
    phase_out = output_dir / "phase_final.nii.gz"
    labels_out = output_dir / "labeled_candidates.nii.gz"
    eroded_out = output_dir / "eroded_candidates.nii.gz"
    
    nib.save(nib.Nifti1Image(t1_norm, t1_nii.affine, t1_nii.header), t1_out)
    nib.save(nib.Nifti1Image(flair_norm, flair_nii.affine, flair_nii.header), flair_out)
    nib.save(nib.Nifti1Image(epi_norm, epi_nii.affine, epi_nii.header), epi_out)
    nib.save(nib.Nifti1Image(phase_norm, phase_nii.affine, phase_nii.header), phase_out)
    nib.save(nib.Nifti1Image(labels, labels_nii.affine, labels_nii.header), labels_out)
    nib.save(nib.Nifti1Image(eroded.astype(np.int16), labels_nii.affine, labels_nii.header), eroded_out)
    
    if verbose:
        print("[Done] Ready for inference.")
    
    return {
        't1': str(t1_out),
        'flair': str(flair_out),
        'epi': str(epi_out),
        'phase': str(phase_out),
        'labeled_candidates': str(labels_out),
        'eroded_candidates': str(eroded_out)
    }