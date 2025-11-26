
"""
make_predictions.py

ALPaCA inference pipeline.
Predicts MS lesions, paramagnetic rim lesions (PRLs), and central vein sign (CVS).

Author: Translated from R package by hufengling
"""

import numpy as np
import torch
import nibabel as nib
import pandas as pd
from pathlib import Path
from typing import Dict, Union, Optional

from ..utils.extract_patch import extract_patch


# Constants
PATCH_SIZE = 24
PATCH_RADIUS = PATCH_SIZE // 2
N_PATCHES_PER_LESION = 20
N_CV_MODELS = 10
N_CONTRASTS = 4      # [T1, FLAIR, Phase, EPI]
N_BIOMARKERS = 3     # [lesion_prob, prl_prob, cvs_prob]
LESIONS_PER_BATCH = 50 


# Probability thresholds for binary classification (Youden's J from training ROC)
# These were optimized on the training set to maximize sensitivity + specificity
THRESHOLDS = {
    'lesion': {
        'youdens_j': 0.5517,    # Balanced:  Sens ≈ 88.6%, Spec ≈ 89.2%
        'specificity': 0.7243,  # High spec: Sens ≈ 69%, Spec ≈ 94%
        'sensitivity': 0.3787   # High sens: Sens ≈ 92%, Spec ≈ 70%
    },
    'prl': {
        'youdens_j': 0.0744,
        'specificity': 0.1135,
        'sensitivity': 0.0350
    },
    'cvs': {
        'youdens_j': 0.2094,
        'specificity': 0.3500,
        'sensitivity': 0.1102
    }
}


# Utilities
def load_nifti(filepath: str) -> np.ndarray:
    """Load NIfTI image and return as numpy array."""
    img = nib.load(filepath)
    return img.get_fdata()


def check_same_shape(images: list, names: list) -> None:
    """Verify all images have the same dimensions."""
    shapes = [img.shape for img in images]
    if len(set(shapes)) != 1:
        raise ValueError(f"Shape mismatch: {dict(zip(names, shapes))}")


def get_valid_patch_centers(labeled_candidates: np.ndarray,
                            candidate_id: int) -> np.ndarray:
    """
    Find all voxel coordinates inside a lesion that can serve as patch centers.
    Valid centers must be at least PATCH_RADIUS voxels from image boundaries.
    """
    candidate_coords = np.argwhere(labeled_candidates == candidate_id)
    dims = labeled_candidates.shape

    valid_mask = (
        (candidate_coords[:, 0] >= PATCH_RADIUS) &
        (candidate_coords[:, 0] < dims[0] - PATCH_RADIUS) &
        (candidate_coords[:, 1] >= PATCH_RADIUS) &
        (candidate_coords[:, 1] < dims[1] - PATCH_RADIUS) &
        (candidate_coords[:, 2] >= PATCH_RADIUS) &
        (candidate_coords[:, 2] < dims[2] - PATCH_RADIUS)
    )

    return candidate_coords[valid_mask]


def make_predictions(
        
    # Input images (file paths or numpy arrays)
    t1: Union[str, np.ndarray],
    flair: Union[str, np.ndarray],
    epi: Union[str, np.ndarray],
    phase: Union[str, np.ndarray],
    labeled_candidates: Union[str, np.ndarray],
    eroded_candidates: Union[str, np.ndarray],

    # Model and output dir
    model_dir: str = None,
    output_dir: str = None,

    # Threshold selection
    lesion_priority: str = 'youdens_j',
    prl_priority: str = 'youdens_j',
    cvs_priority: str = 'youdens_j',

    # Inference options
    n_patches: int = N_PATCHES_PER_LESION,
    n_models: int = N_CV_MODELS,
    clear_discordant: bool = True,
    rotate_patches: bool = True,

    # Optional outputs
    return_probabilities: bool = False,
    save_outputs: bool = True,
    random_seed: Optional[int] = None,
    verbose: bool = True

) -> Dict:
    """
    Run ALPaCA inference to predict Lesion, PRL, and CVS status.

    Args:
        t1, flair, epi, phase: MRI images (file paths or numpy arrays)
        labeled_candidates: Lesion candidates labeled 1, 2, 3, ...
        eroded_candidates: Eroded version of labeled_candidates
        model_dir: Directory containing autoencoder_*.pt and predictor_*.pt files
        output_dir: Where to save results
        lesion_priority: Threshold strategy ('youdens_j', 'specificity', 'sensitivity')
        prl_priority: Threshold strategy for PRL
        cvs_priority: Threshold strategy for CVS
        n_patches: Number of patches to sample per lesion
        n_models: Number of CV models to use (1-10)
        clear_discordant: Clear PRL/CVS predictions when Lesion=0
        rotate_patches: Apply random rotations for test-time augmentation
        return_probabilities: Return full probability maps
        save_outputs: Save results to disk
        random_seed: Seed for reproducibility
        verbose: Print progress

    Returns:
        Dictionary containing:
            - 'mask': Combined segmentation mask (0=background, 1=lesion, 3=lesion+PRL, etc.)
            - 'predictions': DataFrame with per-lesion binary predictions
            - 'probabilities': DataFrame with per-lesion probabilities
            - 'uncertainties': DataFrame with per-lesion standard deviations
            - Optional: 'probability_maps' if return_probabilities=True
    """

    if verbose:
        print("\n┌" + "─"*38 + "┐")
        print("│" + " Lesion Inference ".center(38) + "│")
        print("└" + "─"*38 + "┘")   
    
    if model_dir is None:
        model_dir = Path(__file__).parent.parent.parent / "models"

    if random_seed is not None:
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(random_seed)
        if verbose:
            print(f"Random seed set to: {random_seed}")

    # ========== [STAGE 1] LOAD AND VALIDATE INPUT ========== #
    if verbose:
        print("[1/5] Loading inputs")

    # Validate inputs provided
    if any(x is None for x in [t1, flair, epi, phase, labeled_candidates, eroded_candidates]):
        raise ValueError("All images must be provided")

    # Validate priority choices
    valid_priorities = ['youdens_j', 'specificity', 'sensitivity']
    if lesion_priority not in valid_priorities:
        raise ValueError(f"Invalid lesion_priority: {lesion_priority}")
    if prl_priority not in valid_priorities:
        raise ValueError(f"Invalid prl_priority: {prl_priority}")
    if cvs_priority not in valid_priorities:
        raise ValueError(f"Invalid cvs_priority: {cvs_priority}")

    # Validate numeric parameters
    if n_patches < 1:
        raise ValueError("n_patches must be >= 1")
    if n_models < 1 or n_models > 10:
        raise ValueError("n_models must be between 1 and 10")

    # Load images if paths provided
    ref_img = None  # Store reference for metadata preservation
    if isinstance(t1, str):
        ref_img = nib.load(t1)  # Save reference for affine/header
        t1 = ref_img.get_fdata()
    if isinstance(flair, str):
        flair = load_nifti(flair)
    if isinstance(epi, str):
        epi = load_nifti(epi)
    if isinstance(phase, str):
        phase = load_nifti(phase)
    if isinstance(labeled_candidates, str):
        labeled_candidates = load_nifti(labeled_candidates).astype(np.int32)
    if isinstance(eroded_candidates, str):
        eroded_candidates = load_nifti(eroded_candidates).astype(np.int32)

    check_same_shape(
        [t1, flair, epi, phase, labeled_candidates, eroded_candidates],
        ['t1', 'flair', 'epi', 'phase', 'labeled_candidates', 'eroded_candidates']
    )

    n_lesions = int(labeled_candidates.max())
    if n_lesions == 0:
        if verbose:
            print("No lesion candidates detected.")
        return None

    # ========== [STAGE 2] LOAD MODELS ========== #
    if verbose:
        print(f"[2/5] Loading models")

    model_dir = Path(model_dir)

    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    models_list = []

    for i in range(1, n_models + 1):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        

        autoencoder_path = model_dir / f"autoencoder_{i}.pt"
        predictor_path = model_dir / f"predictor_{i}.pt"

        if not autoencoder_path.exists():
            raise FileNotFoundError(f"Autoencoder model not found: {autoencoder_path}")
        if not predictor_path.exists():
            raise FileNotFoundError(f"Predictor model not found: {predictor_path}")

        try:
            autoencoder = torch.jit.load(str(autoencoder_path), map_location=device)
            predictor = torch.jit.load(str(predictor_path), map_location=device)
        except Exception as e:
            raise RuntimeError(f"Failed to load model {i}: {e}")

        # Extract encoder from autoencoder (R: models_list[[i]][[1]]$encoder)
        encoder = autoencoder.encoder
        encoder.eval()
        predictor.eval()
        models_list.append({'encoder': encoder, 'predictor': predictor})


   # ========== [STAGE 3] BATCH AND INFER ========== #
    if verbose:
        print(f"[3/5] Running inference")

    all_predictions = np.zeros((n_lesions, N_BIOMARKERS)) 
    all_uncertainties = np.zeros((n_lesions, N_BIOMARKERS))
    all_model_disagreement = np.zeros((n_lesions, N_BIOMARKERS))  # Track ensemble disagreement

    n_skipped = 0
    n_batches = (n_lesions + LESIONS_PER_BATCH - 1) // LESIONS_PER_BATCH 

    # Process lesions in batches
    for batch_num, batch_start in enumerate(range(1, n_lesions + 1, LESIONS_PER_BATCH), 1):
        if verbose:
            print(f"  Batch {batch_num}/{n_batches}: Lesions {batch_start}-{min(batch_start + LESIONS_PER_BATCH - 1, n_lesions)}")

        batch_end = min(batch_start + LESIONS_PER_BATCH, n_lesions + 1)
        batch_lesion_ids = range(batch_start, batch_end)
        
        # Extract patches for this batch
        batch_patches_list = []
        batch_lesion_to_patch_idx = []
        current_idx = 0

        for candidate_id in batch_lesion_ids:
            # Find valid patch centers (not on image boundaries)
            valid_coords = get_valid_patch_centers(labeled_candidates, candidate_id)
            n_valid = len(valid_coords)
            
            # Lesion too small or too close to boundary (skip)
            if n_valid == 0: 
                batch_lesion_to_patch_idx.append((current_idx, current_idx))
                n_skipped += 1
                continue
            
            # Sample patch centers (allow replacement if necessary)
            if n_valid < n_patches and rotate_patches:
                n_samples = n_patches
                replace = True
            else:
                n_samples = min(n_patches, n_valid)
                replace = False 
            
            sampled_indices = np.random.choice(n_valid, size=n_samples, replace=replace)
            sampled_centers = valid_coords[sampled_indices]
            
            # Extract patches for this lesion
            for center in sampled_centers:
                starts = center - PATCH_RADIUS
                ends = center + PATCH_RADIUS - 1
                
                patch = extract_patch(
                    candidate_id,
                    starts, ends,
                    t1, flair, epi, phase,
                    labeled_candidates, eroded_candidates,
                    rotate_patches=rotate_patches
                )
                batch_patches_list.append(patch)
            
            # Record patch range for this lesion
            batch_lesion_to_patch_idx.append((current_idx, current_idx + len(sampled_centers)))
            current_idx += len(sampled_centers)

        if len(batch_patches_list) == 0:
            continue

        # Convert to tensor
        batch_patches = torch.stack(batch_patches_list).to(device)

        # Run inference
        batch_model_predictions = []
        with torch.no_grad():
            for model in models_list:
                encoder = model['encoder']
                predictor = model['predictor']
                
                encoded = encoder(batch_patches)
                predictions = predictor(encoded)
                batch_model_predictions.append(predictions)

        # Stack: [n_models, n_patches, n_biomarkers] -> [n_patches, n_models, n_biomarkers]
        batch_model_predictions = torch.stack(batch_model_predictions).transpose(0, 1)

        # Aggregate per patch
        mean_per_patch = batch_model_predictions.mean(dim=1)  
        std_per_patch = batch_model_predictions.std(dim=1)   

        # Map back to lesions
        for i, candidate_id in enumerate(batch_lesion_ids):
            start_idx, end_idx = batch_lesion_to_patch_idx[i]
            n_p = end_idx - start_idx
            
            if n_p == 0:
                # Skipped lesion
                all_predictions[candidate_id - 1] = [0, 0, 0]
                all_uncertainties[candidate_id - 1] = [0, 0, 0]
                all_model_disagreement[candidate_id - 1] = [0, 0, 0]
            else:
                # Get patch-level ensemble means for this lesion
                lesion_patch_means = mean_per_patch[start_idx:end_idx]
                all_predictions[candidate_id - 1] = lesion_patch_means.mean(dim=0).cpu().numpy()
                all_uncertainties[candidate_id - 1] = lesion_patch_means.std(dim=0).cpu().numpy()

                lesion_patch_model_stds = std_per_patch[start_idx:end_idx]
                all_model_disagreement[candidate_id - 1] = lesion_patch_model_stds.mean(dim=0).cpu().numpy()

        # Free memory
        del batch_patches, batch_model_predictions
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    if verbose and n_skipped > 0:
                print(f"  Skipped {n_skipped} lesions (too small or near boundary)")

    # ========== [STAGE 4] THRESHOLD TO BINARY PREDICTIONS ========== #
    if verbose:
        print(f"[4/5] Applying thresholds")

    lesion_thresh = THRESHOLDS['lesion'][lesion_priority]
    prl_thresh = THRESHOLDS['prl'][prl_priority]
    cvs_thresh = THRESHOLDS['cvs'][cvs_priority]

    binary_lesion = (all_predictions[:, 0] > lesion_thresh).astype(int)
    binary_prl = (all_predictions[:, 1] > prl_thresh).astype(int)
    binary_cvs = (all_predictions[:, 2] > cvs_thresh).astype(int)

    if clear_discordant:
        n_discordant_prl = np.sum((binary_prl == 1) & (binary_lesion == 0))
        n_discordant_cvs = np.sum((binary_cvs == 1) & (binary_lesion == 0))

        binary_prl = binary_prl * binary_lesion
        binary_cvs = binary_cvs * binary_lesion

        if verbose and (n_discordant_prl > 0 or n_discordant_cvs > 0):
            print(f"  NOTE: Cleared {n_discordant_prl} discordant PRL, {n_discordant_cvs} discordant CVS predictions")

    # ========== [STAGE 6] CREATE OUTPUT MASKS ========== #
    if verbose:
        print(f"[5/5] Creating output masks")

    # Clever encoding
    lesion_codes = binary_lesion * 1 + binary_prl * 2 + binary_cvs * 4 
    output_mask = np.zeros_like(labeled_candidates, dtype=np.int32)

    for candidate_id in range(1, n_lesions + 1):
        code = lesion_codes[candidate_id - 1]
        output_mask[labeled_candidates == candidate_id] = code

    # Optionally create probability maps
    probability_maps = None
    if return_probabilities:
        lesion_prob_map = np.zeros_like(labeled_candidates, dtype=np.float32)
        prl_prob_map = np.zeros_like(labeled_candidates, dtype=np.float32)
        cvs_prob_map = np.zeros_like(labeled_candidates, dtype=np.float32)

        for candidate_id in range(1, n_lesions + 1):
            mask = (labeled_candidates == candidate_id)
            lesion_prob_map[mask] = all_predictions[candidate_id - 1, 0]
            prl_prob_map[mask] = all_predictions[candidate_id - 1, 1]
            cvs_prob_map[mask] = all_predictions[candidate_id - 1, 2]

        probability_maps = {
            'lesion': lesion_prob_map,
            'prl': prl_prob_map,
            'cvs': cvs_prob_map
        }

    # Create DataFrames
    predictions_df = pd.DataFrame({
        'lesion_id': range(1, n_lesions + 1),
        'lesion': binary_lesion,
        'prl': binary_prl,
        'cvs': binary_cvs
    })

    probabilities_df = pd.DataFrame({
        'lesion_id': range(1, n_lesions + 1),
        'lesion_prob': all_predictions[:, 0],
        'prl_prob': all_predictions[:, 1],
        'cvs_prob': all_predictions[:, 2]
    })

    uncertainties_df = pd.DataFrame({
        'lesion_id': range(1, n_lesions + 1),
        'lesion_std': all_uncertainties[:, 0],
        'prl_std': all_uncertainties[:, 1],
        'cvs_std': all_uncertainties[:, 2]
    })

    disagreements_df = pd.DataFrame({
        'lesion_id': range(1, n_lesions + 1),
        'lesion_disagreement': all_model_disagreement[:, 0],
        'prl_disagreement': all_model_disagreement[:, 1],
        'cvs_disagreement': all_model_disagreement[:, 2]
    })

    # Save outputs
    if save_outputs:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Preserve metadata from reference image if available
        if ref_img is not None:
            affine = ref_img.affine
            header = ref_img.header
        else:
            if verbose:
                print("  WARNING: No reference image - output will use identity affine")
            affine = np.eye(4)
            header = None

        output_mask_nii = nib.Nifti1Image(output_mask, affine=affine, header=header)
        nib.save(output_mask_nii, output_dir / 'alpaca_mask.nii.gz')

        predictions_df.to_csv(output_dir / 'predictions.csv', index=False)
        probabilities_df.to_csv(output_dir / 'probabilities.csv', index=False)
        uncertainties_df.to_csv(output_dir / 'uncertainties.csv', index=False)
        disagreements_df.to_csv(output_dir / 'model_disagreement.csv', index=False)
        
        if probability_maps is not None:
            for name, prob_map in probability_maps.items():
                prob_nii = nib.Nifti1Image(prob_map, affine=affine, header=header)
                nib.save(prob_nii, output_dir / f'{name}_prob.nii.gz')

        if verbose:
            print(f"[Done] Results saved to {output_dir}")

        if verbose:
            print("\n┌" + "─"*38 + "┐")
            print("│" + " Summary ".center(38) + "│")
            print("└" + "─"*38 + "┘")   
            print(f"Total Lesions       : {predictions_df['lesion'].sum()}")
            print(f"Lesions only        : {np.sum(lesion_codes == 1)}")
            print(f"Lesions + PRL       : {np.sum(lesion_codes == 3)}")
            print(f"Lesions + CVS       : {np.sum(lesion_codes == 5)}")
            print(f"Lesions + PRL + CVS : {np.sum(lesion_codes == 7)}")
            print(f"")

    # Return results
    results = {
        'mask': output_mask,
        'predictions': predictions_df,
        'probabilities': probabilities_df,
        'uncertainties': uncertainties_df,
        'disagreements': disagreements_df
    }

    if probability_maps is not None:
        results['probability_maps'] = probability_maps

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run ALPaCA inference')
    parser.add_argument('--t1', required=True, help='Path to T1 image')
    parser.add_argument('--flair', required=True, help='Path to FLAIR image')
    parser.add_argument('--epi', required=True, help='Path to EPI magnitude image')
    parser.add_argument('--phase', required=True, help='Path to EPI phase image')
    parser.add_argument('--labeled-candidates', required=True, help='Path to labeled candidates')
    parser.add_argument('--eroded-candidates', required=True, help='Path to eroded candidates')
    parser.add_argument('--model-dir', required=True, help='Directory with model weights')
    parser.add_argument('--output-dir', required=True, help='Output directory')
    parser.add_argument('--lesion-priority', default='youdens_j',
                       choices=['youdens_j', 'specificity', 'sensitivity'])
    parser.add_argument('--prl-priority', default='youdens_j',
                       choices=['youdens_j', 'specificity', 'sensitivity'])
    parser.add_argument('--cvs-priority', default='youdens_j',
                       choices=['youdens_j', 'specificity', 'sensitivity'])
    parser.add_argument('--n-patches', type=int, default=20)
    parser.add_argument('--n-models', type=int, default=10)
    parser.add_argument('--rotate-patches', action='store_true')
    parser.add_argument('--return-probabilities', action='store_true')

    args = parser.parse_args()

    results = make_predictions(
        t1=args.t1,
        flair=args.flair,
        epi=args.epi,
        phase=args.phase,
        labeled_candidates=args.labeled_candidates,
        eroded_candidates=args.eroded_candidates,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        lesion_priority=args.lesion_priority,
        prl_priority=args.prl_priority,
        cvs_priority=args.cvs_priority,
        n_patches=args.n_patches,
        n_models=args.n_models,
        rotate_patches=args.rotate_patches,
        return_probabilities=args.return_probabilities
    )
