
# extract_patch.py

import numpy as np
import torch

from .rotate_patch import rotate_patch 

def extract_patch(candidate_id, patch_starts, patch_ends,
                  t1, flair, epi, phase,
                  labeled_candidates, eroded_candidates,
                  rotate_patches=False):
    """
    Extract 24x24x24 patch around a lesion candidate with attention masking.

    Attention masking strategy:
    - For T1/FLAIR/Phase: Dims other lesions to 10%, keeps target + background at 100%
    - For EPI: Zeros out everything except target, emphasizes eroded core (CVS detection)

    Args:
        candidate_id: Integer label for target lesion
        patch_starts: [x, y, z] start coordinates (inclusive)
        patch_ends: [x, y, z] end coordinates (inclusive)
        t1, flair, epi, phase: 3D numpy arrays (same shape)
        labeled_candidates: 3D array with integer labels for each lesion
        eroded_candidates: 3D array with eroded lesion labels
        rotate_patches: Whether to apply random rotation

    Returns:
        torch.Tensor of shape [4, 24, 24, 24]
    """

    # Extract patch regions (numpy not inclusive of end)
    s = patch_starts
    e = patch_ends
    
    lesion_mask = labeled_candidates[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
    eroded_mask = eroded_candidates[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]

    # Isolation mask for T1/FLAIR/Phase
    # Dims other candidates to 10%, keeps background + target at 100%
    is_background = (lesion_mask == 0)
    is_target = (lesion_mask == candidate_id)
    isolation_mask = 0.1 + 0.9 * (is_background | is_target).astype(np.float32)

    # EPI isolation mask (zeros out non-target, emphasizes interior for CVS)
    epi_mask = ((lesion_mask == candidate_id).astype(np.float32) +
                (eroded_mask == candidate_id).astype(np.float32))
    
    # Extract and mask patches
    t1_patch = t1[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1] * isolation_mask
    flair_patch = flair[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1] * isolation_mask
    phase_patch = phase[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1] * isolation_mask
    epi_patch = epi[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1] * epi_mask
    
    patches = [
        torch.from_numpy(t1_patch).float(),
        torch.from_numpy(flair_patch).float(),
        torch.from_numpy(phase_patch).float(),
        torch.from_numpy(epi_patch).float()
    ]


    # Optional rotation
    if rotate_patches:
        invert = np.random.randint(0, 2)        # Mirror patch
        face = np.random.randint(1, 7)          # Which face points down
        rotations = np.random.randint(0, 4)     # Rotate once facing down
        patches = [rotate_patch(p, invert, face, rotations) for p in patches]

    # Add channel dimension and concatenate
    patches = [p.unsqueeze(0) for p in patches] # [1, 24, 24, 24]
    return torch.cat(patches, dim=0)            # [4, 24, 24, 24]
