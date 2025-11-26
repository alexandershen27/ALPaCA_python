"""
Simple visualization script for ALPaCA presentation
Generates figures showing patches and attention masking
"""

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import pandas as pd

# Try to import extract_patch - add path if needed
try:
    from alpaca.utils.extract_patch import extract_patch
except ImportError:
    # If alpaca not in path, look for it relative to script or ask user
    script_dir = Path(__file__).parent
    possible_paths = [
        script_dir.parent,  # If script is in ALPaCA_python/scripts/
        script_dir.parent / 'ALPaCA_python',  # If script is in alpaca/
        Path('/home/shena2/alpaca/ALPaCA_python'),  # Your specific path
        Path.cwd(),  # Current directory
    ]
    
    alpaca_found = False
    for path in possible_paths:
        if (path / 'alpaca' / 'utils' / 'extract_patch.py').exists():
            sys.path.insert(0, str(path))
            from alpaca.utils.extract_patch import extract_patch
            alpaca_found = True
            print(f"Found alpaca package at: {path}")
            break
    
    if not alpaca_found:
        print("ERROR: Could not find alpaca package!")
        print("Please run this script from ALPaCA_python directory, or:")
        print("  python visualize_patches.py <subject_dir> --alpaca-root /path/to/ALPaCA_python")
        sys.exit(1)


def compute_attention_masks(lesion_mask, eroded_mask, candidate_id):
    """
    Compute attention masks exactly as used in extract_patch.py
    
    Isolation mask (for T1/FLAIR/Phase):
        - Background + target lesion: 1.0 (100%)
        - Other lesions: 0.1 (10%, dimmed)
    
    EPI mask (for EPI magnitude):
        - Background/other lesions: 0 (zeroed out)
        - Target lesion surface only: 1 (normal intensity)
        - Target lesion surface + eroded core: 2 (2x boost for CVS detection)
    
    Args:
        lesion_mask: 3D array with lesion labels
        eroded_mask: 3D array with eroded lesion labels
        candidate_id: ID of target lesion
    
    Returns:
        isolation_mask: Float array [0.1 or 1.0]
        epi_mask: Float array [0, 1, or 2]
    """
    is_background = (lesion_mask == 0)
    is_target = (lesion_mask == candidate_id)
    isolation_mask = 0.1 + 0.9 * (is_background | is_target).astype(np.float32)
    
    # CRITICAL: Cast to float BEFORE adding to get arithmetic (not boolean OR)
    # Wrong: (bool + bool).astype(float) → boolean OR → [0, 1]
    # Right: bool.astype(float) + bool.astype(float) → arithmetic → [0, 1, 2]
    epi_mask = ((lesion_mask == candidate_id).astype(np.float32) + 
                (eroded_mask == candidate_id).astype(np.float32))
    
    return isolation_mask, epi_mask


def load_subject_data(subject_dir, from_alpaca_output=False):
    """Load preprocessed images for a subject
    
    Args:
        subject_dir: Path to directory with images
        from_alpaca_output: If True, expects structure with preprocessed/ subfolder
    """
    subject_dir = Path(subject_dir)
    
    if from_alpaca_output:
        # ALPaCA output structure: results/test/preprocessed/*.nii.gz
        img_dir = subject_dir / 'preprocessed'
    else:
        # Direct path to preprocessed images
        img_dir = subject_dir
    
    t1 = nib.load(img_dir / 't1_final.nii.gz').get_fdata()
    flair = nib.load(img_dir / 'flair_final.nii.gz').get_fdata()
    epi = nib.load(img_dir / 'epi_final.nii.gz').get_fdata()
    phase = nib.load(img_dir / 'phase_final.nii.gz').get_fdata()
    labeled = nib.load(img_dir / 'labeled_candidates.nii.gz').get_fdata().astype(np.int32)
    eroded = nib.load(img_dir / 'eroded_candidates.nii.gz').get_fdata().astype(np.int32)
    
    return t1, flair, epi, phase, labeled, eroded


def load_predictions(output_dir):
    """Load ALPaCA predictions from output directory
    
    Args:
        output_dir: Path to directory containing predictions/ subfolder
    
    Returns:
        predictions_df, probabilities_df (or None if not found)
    """
    output_dir = Path(output_dir)
    pred_dir = output_dir / 'predictions'
    
    pred_file = pred_dir / 'predictions.csv'
    prob_file = pred_dir / 'probabilities.csv'
    
    if not pred_file.exists():
        return None, None
    
    predictions = pd.read_csv(pred_file)
    probabilities = pd.read_csv(prob_file) if prob_file.exists() else None
    
    return predictions, probabilities


def filter_lesions_by_criteria(predictions, require_prl=False, require_cvs=False):
    """Filter lesions based on prediction criteria (strict AND logic)
    
    Args:
        predictions: DataFrame with predictions
        require_prl: Only include PRL=1 lesions
        require_cvs: Only include CVS=1 lesions
    
    Returns:
        Array of lesion_ids that meet ALL criteria
    """
    mask = (predictions['lesion'] == 1)  # Must be predicted as lesion
    
    if require_prl:
        mask &= (predictions['prl'] == 1)
    
    if require_cvs:
        mask &= (predictions['cvs'] == 1)
    
    filtered = predictions[mask]
    
    return filtered['lesion_id'].values, filtered


def find_good_lesion(labeled_candidates, min_size=50, random_seed=None, 
                     prefer_multiple=False, allowed_lesion_ids=None):
    """Find a reasonably-sized lesion, optionally with other lesions nearby
    
    Args:
        labeled_candidates: 3D array with lesion labels
        min_size: Minimum lesion size in voxels
        random_seed: Random seed for selection
        prefer_multiple: Prefer patches with multiple lesions
        allowed_lesion_ids: If provided, only consider these lesion IDs
    """
    lesion_ids = np.unique(labeled_candidates)
    lesion_ids = lesion_ids[lesion_ids > 0]
    
    # Filter by allowed IDs if provided
    if allowed_lesion_ids is not None:
        lesion_ids = np.array([lid for lid in lesion_ids if lid in allowed_lesion_ids])
        if len(lesion_ids) == 0:
            print("  ERROR: No lesions match the specified criteria!")
            return None, None
    
    # Filter by size
    valid_lesions = []
    for lesion_id in lesion_ids:
        coords = np.argwhere(labeled_candidates == lesion_id)
        size = len(coords)
        
        if size >= min_size:
            center = coords.mean(axis=0).astype(int)
            valid_lesions.append((lesion_id, center, size))
    
    if not valid_lesions:
        return None, None
    
    # If we want multiple lesions, filter to only those with neighbors
    if prefer_multiple:
        PATCH_RADIUS = 12
        multi_lesion_candidates = []
        
        for lesion_id, center, size in valid_lesions:
            # Count how many other lesions are within patch radius
            nearby_count = 0
            for other_id, other_center, _ in valid_lesions:
                if other_id != lesion_id:
                    dist = np.linalg.norm(center - other_center)
                    # FIX: Use PATCH_RADIUS, not PATCH_RADIUS*2, so lesion is actually in patch
                    if dist < PATCH_RADIUS:  # Actually within patch
                        nearby_count += 1
            
            if nearby_count > 0:
                multi_lesion_candidates.append((lesion_id, center, size, nearby_count))
        
        if not multi_lesion_candidates:
            print("  WARNING: No multi-lesion patches found! Using single lesion instead.")
        else:
            # Now pick from multi-lesion candidates (randomly if seed provided)
            if random_seed is not None:
                np.random.seed(random_seed)
                lesion_id, center, size, nearby = multi_lesion_candidates[
                    np.random.randint(len(multi_lesion_candidates))
                ]
                print(f"  Randomly selected lesion {lesion_id} with {nearby} nearby lesion(s) (size={size}, seed={random_seed})")
            else:
                # Pick the one with most neighbors
                lesion_id, center, size, nearby = max(multi_lesion_candidates, key=lambda x: x[3])
                print(f"  Selected lesion {lesion_id} with {nearby} nearby lesion(s) (size={size})")
            
            return lesion_id, center
    
    # Standard selection (not multi-lesion)
    if random_seed is not None:
        # Pick random lesion from valid ones
        np.random.seed(random_seed)
        lesion_id, center, size = valid_lesions[np.random.randint(len(valid_lesions))]
        print(f"  Randomly selected lesion {lesion_id} (size={size}, seed={random_seed})")
    else:
        # Pick lesion closest to brain center
        brain_center = np.array(labeled_candidates.shape) // 2
        best_lesion = None
        best_distance = float('inf')
        
        for lesion_id, center, size in valid_lesions:
            distance = np.linalg.norm(center - brain_center)
            if distance < best_distance:
                best_distance = distance
                best_lesion = lesion_id
                best_center = center
                best_size = size
        
        lesion_id = best_lesion
        center = best_center
        size = best_size
        print(f"  Selected center lesion {lesion_id} (size={size})")
    
    return lesion_id, center


def adjust_center_for_multi_lesion(labeled_candidates, target_lesion_id, target_center):
    """Adjust patch center to better capture nearby lesions
    
    Moves the camera towards the nearest lesion to show both better
    
    Args:
        labeled_candidates: 3D array with lesion labels
        target_lesion_id: The main lesion ID
        target_center: Original center of target lesion
    
    Returns:
        Adjusted center coordinates
    """
    PATCH_RADIUS = 12
    
    # Find all lesions
    lesion_ids = np.unique(labeled_candidates)
    lesion_ids = lesion_ids[(lesion_ids > 0) & (lesion_ids != target_lesion_id)]
    
    if len(lesion_ids) == 0:
        return target_center
    
    # Find nearest lesion
    nearest_lesion = None
    nearest_dist = float('inf')
    
    for lesion_id in lesion_ids:
        coords = np.argwhere(labeled_candidates == lesion_id)
        other_center = coords.mean(axis=0)
        dist = np.linalg.norm(target_center - other_center)
        
        if dist < nearest_dist and dist < PATCH_RADIUS:
            nearest_dist = dist
            nearest_lesion = lesion_id
            nearest_center = other_center
    
    if nearest_lesion is None:
        return target_center
    
    # Move patch center towards the nearby lesion (halfway between)
    adjusted_center = ((target_center + nearest_center) / 2).astype(int)
    
    print(f"  Adjusted patch center towards lesion {nearest_lesion} (distance={nearest_dist:.1f} voxels)")
    
    return adjusted_center


def visualize_full_slice(t1, flair, epi, phase, labeled, slice_idx, save_path=None, save_individual=False):
    """Show all 5 modalities at the same axial slice"""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))
    
    axes[0].imshow(t1[:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[0].set_title('T1')
    axes[0].axis('off')
    
    axes[1].imshow(flair[:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[1].set_title('FLAIR')
    axes[1].axis('off')
    
    axes[2].imshow(epi[:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[2].set_title('EPI Magnitude')
    axes[2].axis('off')
    
    axes[3].imshow(phase[:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[3].set_title('EPI Phase')
    axes[3].axis('off')
    
    # Lesion mask overlay
    axes[4].imshow(t1[:, :, slice_idx].T, cmap='gray', origin='lower')
    lesion_slice = labeled[:, :, slice_idx].T
    axes[4].imshow(np.ma.masked_where(lesion_slice == 0, lesion_slice), 
                   cmap='hot', alpha=0.5, origin='lower')
    axes[4].set_title('Lesion Candidates')
    axes[4].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        # Save individual images if requested
        if save_individual:
            save_dir = Path(save_path).parent / 'individual'
            save_dir.mkdir(exist_ok=True)
            
            # Save each modality separately
            for idx, (data, name) in enumerate([
                (t1[:, :, slice_idx].T, 't1'),
                (flair[:, :, slice_idx].T, 'flair'),
                (epi[:, :, slice_idx].T, 'epi_mag'),
                (phase[:, :, slice_idx].T, 'epi_phase')
            ]):
                fig_single = plt.figure(figsize=(4, 4))
                plt.imshow(data, cmap='gray', origin='lower')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(save_dir / f'{name}_slice.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            # Save lesion overlay
            fig_single = plt.figure(figsize=(4, 4))
            plt.imshow(t1[:, :, slice_idx].T, cmap='gray', origin='lower')
            plt.imshow(np.ma.masked_where(lesion_slice == 0, lesion_slice), 
                      cmap='hot', alpha=0.5, origin='lower')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(save_dir / 'lesion_overlay.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  Individual images saved to {save_dir}")
    else:
        plt.show()
    
    plt.close()


def visualize_patch_3d(patch_tensor, save_path=None, save_individual=False):
    """Visualize center slice of each modality in a 4-channel patch"""
    # patch_tensor is [4, 24, 24, 24]
    patch_np = patch_tensor.cpu().numpy()
    center_slice = 12  # Middle of 24
    
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    titles = ['T1 Patch', 'FLAIR Patch', 'Phase Patch', 'EPI Patch']
    
    for i in range(4):
        axes[i].imshow(patch_np[i, :, :, center_slice].T, cmap='gray', origin='lower')
        axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        
        # Save individual patches if requested
        if save_individual:
            save_dir = Path(save_path).parent / 'individual'
            save_dir.mkdir(exist_ok=True)
            
            names = ['t1_patch', 'flair_patch', 'phase_patch', 'epi_patch']
            for i, name in enumerate(names):
                fig_single = plt.figure(figsize=(4, 4))
                plt.imshow(patch_np[i, :, :, center_slice].T, cmap='gray', origin='lower')
                plt.axis('off')
                plt.tight_layout()
                plt.savefig(save_dir / f'{name}.png', dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"  Individual patches saved to {save_dir}")
    else:
        plt.show()
    
    plt.close()


def interactive_browse(t1, flair, epi, phase, labeled, eroded, 
                       output_dir, predictions_df=None, probabilities_df=None,
                       allowed_lesion_ids=None):
    """Interactive mode to browse lesions and save good ones
    
    Controls:
        h/j/k/l - Move camera left/down/up/right
        n - Next lesion
        p - Previous lesion  
        s - Save current view
        q - Quit
    """
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Find all valid lesions
    lesion_ids = np.unique(labeled)
    lesion_ids = lesion_ids[lesion_ids > 0]
    
    if allowed_lesion_ids is not None:
        lesion_ids = np.array([lid for lid in lesion_ids if lid in allowed_lesion_ids])
    
    # Get centers and filter by size
    candidates = []
    for lesion_id in lesion_ids:
        coords = np.argwhere(labeled == lesion_id)
        if len(coords) >= 50:  # Min size
            center = coords.mean(axis=0).astype(int)
            candidates.append((lesion_id, center))
    
    if len(candidates) == 0:
        print("No valid lesions found!")
        return
    
    print(f"\n{'='*60}")
    print(f"INTERACTIVE BROWSER - {len(candidates)} lesions")
    print(f"{'='*60}")
    print("Controls:")
    print("  h/j/k/l = Move camera left/down/up/right (2 voxels)")
    print("  n/p     = Next/Previous lesion")
    print("  s       = Save current view")
    print("  q       = Quit")
    print(f"{'='*60}\n")
    
    # State
    state = {
        'index': 0,
        'offset': np.array([0, 0, 0]),
        'saved_count': 0
    }
    
    PATCH_RADIUS = 12
    MOVE_STEP = 2
    
    def get_current_patch():
        """Get current lesion and patch coordinates"""
        lesion_id, base_center = candidates[state['index']]
        center = base_center + state['offset']
        
        # Clamp to valid range
        center = np.clip(center, PATCH_RADIUS, np.array(labeled.shape) - PATCH_RADIUS - 1)
        
        starts = center - PATCH_RADIUS
        ends = center + PATCH_RADIUS - 1
        
        return lesion_id, center, starts, ends
    
    def render():
        """Render current view"""
        lesion_id, center, starts, ends = get_current_patch()
        s, e = starts, ends
        
        # Get prediction info
        pred_text = f"Lesion {lesion_id}"
        if predictions_df is not None:
            pred = predictions_df[predictions_df['lesion_id'] == lesion_id].iloc[0]
            tags = []
            if pred['lesion'] == 1:
                tags.append("✓Lesion")
            if pred['prl'] == 1:
                tags.append("✓PRL")
            if pred['cvs'] == 1:
                tags.append("✓CVS")
            if tags:
                pred_text += " | " + " ".join(tags)
            
            if probabilities_df is not None:
                prob = probabilities_df[probabilities_df['lesion_id'] == lesion_id].iloc[0]
                pred_text += f" | P(L/PRL/CVS)={prob['lesion_prob']:.2f}/{prob['prl_prob']:.2f}/{prob['cvs_prob']:.2f}"
        
        # Extract patches
        center_slice = center[2]
        slice_idx = center_slice - s[2]
        
        t1_patch = t1[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
        flair_patch = flair[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
        epi_patch = epi[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
        phase_patch = phase[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
        lesion_patch = labeled[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
        
        # Clear and redraw
        for ax in axes:
            ax.clear()
        
        axes[0].imshow(t1_patch[:, :, slice_idx].T, cmap='gray', origin='lower')
        axes[0].set_title('T1')
        axes[0].axis('off')
        
        axes[1].imshow(flair_patch[:, :, slice_idx].T, cmap='gray', origin='lower')
        axes[1].set_title('FLAIR')
        axes[1].axis('off')
        
        axes[2].imshow(phase_patch[:, :, slice_idx].T, cmap='gray', origin='lower')
        axes[2].set_title('Phase')
        axes[2].axis('off')
        
        axes[3].imshow(epi_patch[:, :, slice_idx].T, cmap='gray', origin='lower')
        axes[3].set_title('EPI')
        axes[3].axis('off')
        
        # Draw lesion boundaries
        lesion_slice = lesion_patch[:, :, slice_idx].T
        for lid in np.unique(lesion_slice):
            if lid > 0:
                mask = (lesion_slice == lid).astype(float)
                color = 'red' if lid == lesion_id else 'yellow'
                alpha = 0.7 if lid == lesion_id else 0.4
                for ax in axes:
                    ax.contour(mask, levels=[0.5], colors=color, linewidths=2, alpha=alpha)
        
        # Update title
        offset_str = f"offset=({state['offset'][0]:+d},{state['offset'][1]:+d},{state['offset'][2]:+d})"
        fig.suptitle(f"[{state['index']+1}/{len(candidates)}] {pred_text} | {offset_str}", 
                     fontsize=11, fontweight='bold')
        
        plt.draw()
    
    def save_current():
        """Save current view"""
        lesion_id, center, starts, ends = get_current_patch()
        
        # Generate all figures for this lesion
        save_num = state['saved_count'] + 1
        save_dir = output_dir / f'saved_{save_num:03d}_lesion_{lesion_id}'
        save_dir.mkdir(exist_ok=True)
        
        print(f"\n  Saving view {save_num} (lesion {lesion_id}) to {save_dir}...")
        
        # Save full slice
        visualize_full_slice(t1, flair, epi, phase, labeled, 
                           center[2],
                           save_path=save_dir / 'fig1_full_slice.png',
                           save_individual=True)
        
        # Save patch
        patch_tensor = extract_patch(
            lesion_id, starts, ends,
            t1, flair, epi, phase,
            labeled, eroded,
            rotate_patches=False
        )
        
        visualize_patch_3d(patch_tensor, 
                          save_path=save_dir / 'fig2_patch_3d.png',
                          save_individual=True)
        
        visualize_attention_mask(labeled, eroded, lesion_id, starts, ends,
                                save_path=save_dir / 'fig3_attention_masks.png')
        
        visualize_masked_comparison(t1, flair, epi, phase, labeled, eroded,
                                   lesion_id, starts, ends,
                                   save_path=save_dir / 'fig4_masking_effect.png')
        
        if predictions_df is not None:
            visualize_predictions_annotated(t1, flair, epi, phase, labeled,
                                          lesion_id, starts, ends,
                                          predictions_df, probabilities_df,
                                          save_path=save_dir / 'fig5_predictions_annotated.png')
        
        state['saved_count'] += 1
        print(f"  ✓ Saved! ({state['saved_count']} total)")
        print(f"  📝 Debug log: {save_dir / 'masking_debug.log'}")
    
    def on_key(event):
        """Handle keyboard input"""
        if event.key == 'q':
            print(f"\nQuitting... Saved {state['saved_count']} views.")
            plt.close()
            return
        
        elif event.key == 'n':  # Next lesion
            state['index'] = (state['index'] + 1) % len(candidates)
            state['offset'] = np.array([0, 0, 0])
            render()
        
        elif event.key == 'p':  # Previous lesion
            state['index'] = (state['index'] - 1) % len(candidates)
            state['offset'] = np.array([0, 0, 0])
            render()
        
        elif event.key == 'h':  # Move left
            state['offset'][0] -= MOVE_STEP
            render()
        
        elif event.key == 'l':  # Move right
            state['offset'][0] += MOVE_STEP
            render()
        
        elif event.key == 'j':  # Move down
            state['offset'][1] -= MOVE_STEP
            render()
        
        elif event.key == 'k':  # Move up
            state['offset'][1] += MOVE_STEP
            render()
        
        elif event.key == 's':  # Save
            save_current()
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Initial render
    render()
    plt.show()


def visualize_attention_mask(labeled, eroded, candidate_id, starts, ends, save_path=None):
    """Show the attention masks for isolation and EPI"""
    s = starts
    e = ends
    center_slice = (e[2] + s[2]) // 2
    
    # Get patch regions
    lesion_mask = labeled[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
    eroded_mask = eroded[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
    
    # Use centralized mask calculation
    isolation_mask, epi_mask = compute_attention_masks(lesion_mask, eroded_mask, candidate_id)
    
    # Show center slice
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    slice_idx = center_slice - s[2]
    
    axes[0].imshow(lesion_mask[:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[0].set_title('Lesion Candidates\n(in patch)')
    axes[0].axis('off')
    
    axes[1].imshow(isolation_mask[:, :, slice_idx].T, cmap='viridis', origin='lower', vmin=0, vmax=1)
    axes[1].set_title('Isolation Mask\n(T1/FLAIR/Phase)')
    axes[1].axis('off')
    
    axes[2].imshow(epi_mask[:, :, slice_idx].T, cmap='viridis', origin='lower', vmin=0, vmax=2)
    axes[2].set_title('EPI Mask\n(emphasizes core)')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_predictions_annotated(t1, flair, epi, phase, labeled, 
                                    candidate_id, starts, ends,
                                    predictions_df, probabilities_df,
                                    save_path=None):
    """Show patches with prediction annotations for inspection"""
    s = starts
    e = ends
    center_slice = (e[2] + s[2]) // 2
    slice_idx = center_slice - s[2]
    
    # Get patch region
    lesion_mask = labeled[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
    
    # Find all lesions in patch
    lesions_in_patch = np.unique(lesion_mask)
    lesions_in_patch = lesions_in_patch[lesions_in_patch > 0]
    
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    
    # Get predictions for target lesion
    target_pred = predictions_df[predictions_df['lesion_id'] == candidate_id].iloc[0]
    target_prob = probabilities_df[probabilities_df['lesion_id'] == candidate_id].iloc[0] if probabilities_df is not None else None
    
    # Build title with predictions
    title_parts = [f"Target Lesion {candidate_id}"]
    if target_pred['lesion'] == 1:
        title_parts.append("✓ Lesion")
    if target_pred['prl'] == 1:
        title_parts.append("✓ PRL")
    if target_pred['cvs'] == 1:
        title_parts.append("✓ CVS")
    
    if target_prob is not None:
        prob_text = f"(P={target_prob['lesion_prob']:.2f}, PRL={target_prob['prl_prob']:.2f}, CVS={target_prob['cvs_prob']:.2f})"
        title_parts.append(prob_text)
    
    main_title = " | ".join(title_parts)
    
    # Show patches
    axes[0].imshow(t1[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1][:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[0].set_title('T1')
    axes[0].axis('off')
    
    axes[1].imshow(flair[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1][:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[1].set_title('FLAIR')
    axes[1].axis('off')
    
    axes[2].imshow(phase[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1][:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[2].set_title('Phase')
    axes[2].axis('off')
    
    axes[3].imshow(epi[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1][:, :, slice_idx].T, cmap='gray', origin='lower')
    axes[3].set_title('EPI')
    axes[3].axis('off')
    
    # Overlay lesion boundaries on all
    for ax in axes:
        lesion_slice = lesion_mask[:, :, slice_idx].T
        # Draw contours for all lesions
        for lesion_id in lesions_in_patch:
            mask = (lesion_slice == lesion_id).astype(float)
            color = 'red' if lesion_id == candidate_id else 'yellow'
            alpha = 0.5 if lesion_id == candidate_id else 0.3
            ax.contour(mask, levels=[0.5], colors=color, linewidths=2, alpha=alpha)
    
    fig.suptitle(main_title, fontsize=12, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    else:
        plt.show()
    
    plt.close()


def visualize_masked_comparison(t1, flair, epi, phase, labeled, eroded, 
                                candidate_id, starts, ends, save_path=None):
    """Show the attention masking effect: raw patches -> masks -> final result"""
    s = starts
    e = ends
    center_slice = (e[2] + s[2]) // 2
    slice_idx = center_slice - s[2]
    
    # Set up logging
    if save_path:
        log_file = Path(save_path).parent / 'masking_debug.log'
    else:
        log_file = 'masking_debug.log'
    
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"MASKING VISUALIZATION - Candidate {candidate_id}\n")
        f.write(f"{'='*80}\n")
        f.write(f"Patch bounds: starts={s}, ends={e}\n")
        f.write(f"Center slice Z-coordinate: {center_slice} (local slice index: {slice_idx})\n")
    
    # Extract raw patches (BEFORE masking)
    t1_raw = t1[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
    flair_raw = flair[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
    phase_raw = phase[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
    epi_raw = epi[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
    
    # Get lesion masks
    lesion_mask = labeled[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
    eroded_mask = eroded[s[0]:e[0]+1, s[1]:e[1]+1, s[2]:e[2]+1]
    
    with open(log_file, 'a') as f:
        f.write(f"\n--- 3D PATCH ANALYSIS (all 24 slices) ---\n")
        f.write(f"Unique lesion IDs in 3D patch: {np.unique(lesion_mask)}\n")
        f.write(f"Target candidate ID: {candidate_id}\n")
        f.write(f"Total voxels with target ID: {np.sum(lesion_mask == candidate_id)}\n")
        
        for lid in np.unique(lesion_mask):
            if lid > 0:
                count = np.sum(lesion_mask == lid)
                f.write(f"  Lesion {lid}: {count} voxels total\n")
        
        f.write(f"\nEroded mask (3D patch):\n")
        f.write(f"Unique eroded IDs: {np.unique(eroded_mask)}\n")
        f.write(f"Voxels with eroded target ID: {np.sum(eroded_mask == candidate_id)}\n")
        
        for eid in np.unique(eroded_mask):
            if eid > 0:
                count = np.sum(eroded_mask == eid)
                f.write(f"  Eroded lesion {eid}: {count} voxels total\n")
        
        f.write(f"\n--- SINGLE SLICE ANALYSIS (Z={center_slice}, slice index {slice_idx}) ---\n")
        lesion_slice = lesion_mask[:, :, slice_idx]
        eroded_slice = eroded_mask[:, :, slice_idx]
        
        f.write(f"Lesion IDs visible in this slice: {np.unique(lesion_slice)}\n")
        for lid in np.unique(lesion_slice):
            if lid > 0:
                count = np.sum(lesion_slice == lid)
                f.write(f"  Lesion {lid}: {count} voxels in this slice\n")
        
        f.write(f"\nEroded IDs visible in this slice: {np.unique(eroded_slice)}\n")
        for eid in np.unique(eroded_slice):
            if eid > 0:
                count = np.sum(eroded_slice == eid)
                f.write(f"  Eroded lesion {eid}: {count} voxels in this slice\n")
    
    # Use centralized mask calculation
    isolation_mask, epi_mask = compute_attention_masks(lesion_mask, eroded_mask, candidate_id)
    
    with open(log_file, 'a') as f:
        f.write(f"\n--- MASK VALUES (center slice) ---\n")
        iso_slice = isolation_mask[:, :, slice_idx]
        epi_slice = epi_mask[:, :, slice_idx]
        
        f.write(f"Isolation mask unique values: {np.unique(iso_slice)}\n")
        f.write(f"  Background (1.0): {np.sum(iso_slice == 1.0)} voxels\n")
        f.write(f"  Other lesions (0.1): {np.sum(iso_slice == 0.1)} voxels\n")
        
        f.write(f"\nEPI mask unique values: {np.unique(epi_slice)}\n")
        f.write(f"  Background/other (0): {np.sum(epi_slice == 0)} voxels\n")
        f.write(f"  Surface only (1): {np.sum(epi_slice == 1)} voxels\n")
        f.write(f"  Surface + eroded core (2): {np.sum(epi_slice == 2)} voxels\n")
        f.write(f"{'='*80}\n")
    
    # Apply masks (AFTER masking)
    t1_masked = t1_raw * isolation_mask
    flair_masked = flair_raw * isolation_mask
    phase_masked = phase_raw * isolation_mask
    epi_masked = epi_raw * epi_mask
    
    # Calculate intensity ranges from RAW images for consistent scaling
    t1_vmin, t1_vmax = t1_raw.min(), t1_raw.max()
    flair_vmin, flair_vmax = flair_raw.min(), flair_raw.max()
    phase_vmin, phase_vmax = phase_raw.min(), phase_raw.max()
    epi_vmin, epi_vmax = epi_raw.min(), epi_raw.max()
    epi_masked_vmax = epi_vmax * 2  # For 2x boost
    
    # Create figure: 3 rows x 4 columns
    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    
    # Row 1: Raw patches
    axes[0, 0].imshow(t1_raw[:, :, slice_idx].T, cmap='gray', origin='lower', vmin=t1_vmin, vmax=t1_vmax)
    axes[0, 0].set_title('T1', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(flair_raw[:, :, slice_idx].T, cmap='gray', origin='lower', vmin=flair_vmin, vmax=flair_vmax)
    axes[0, 1].set_title('FLAIR', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    axes[0, 2].imshow(phase_raw[:, :, slice_idx].T, cmap='gray', origin='lower', vmin=phase_vmin, vmax=phase_vmax)
    axes[0, 2].set_title('EPI Phase', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    axes[0, 3].imshow(epi_raw[:, :, slice_idx].T, cmap='gray', origin='lower', vmin=epi_vmin, vmax=epi_vmax)
    axes[0, 3].set_title('EPI Magnitude', fontsize=14, fontweight='bold')
    axes[0, 3].axis('off')
    
    # Row 2: Attention masks
    im1 = axes[1, 0].imshow(isolation_mask[:, :, slice_idx].T, cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
    axes[1, 0].set_title('Isolation Mask', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    plt.colorbar(im1, ax=axes[1, 0], fraction=0.046, pad=0.04, label='0.1 / 1.0')
    
    axes[1, 1].text(0.5, 0.5, 'Applied to:\nT1, FLAIR, Phase', 
                    ha='center', va='center', fontsize=12, transform=axes[1, 1].transAxes)
    axes[1, 1].axis('off')
    
    # Use discrete colormap for EPI mask
    from matplotlib.colors import ListedColormap
    epi_cmap = ListedColormap(['darkred', 'yellow', 'lime'])
    im2 = axes[1, 2].imshow(epi_mask[:, :, slice_idx].T, cmap=epi_cmap, origin='lower', vmin=0, vmax=2)
    axes[1, 2].set_title('EPI Mask', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04, ticks=[0, 1, 2])
    cbar2.ax.set_yticklabels(['0 (off)', '1x', '2x'])
    
    axes[1, 3].text(0.5, 0.5, 'Applied to:\nEPI Magnitude', 
                    ha='center', va='center', fontsize=12, transform=axes[1, 3].transAxes)
    axes[1, 3].axis('off')
    
    # Save individual mask images if save_path is provided
    if save_path:
        mask_dir = Path(save_path).parent / 'individual'
        mask_dir.mkdir(exist_ok=True)
        
        # Save isolation mask
        fig_iso = plt.figure(figsize=(6, 6))
        plt.imshow(isolation_mask[:, :, slice_idx].T, cmap='RdYlGn', origin='lower', vmin=0, vmax=1)
        plt.colorbar(label='0.1 / 1.0')
        plt.title('Isolation Mask')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(mask_dir / 'isolation_mask.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save EPI mask
        fig_epi = plt.figure(figsize=(6, 6))
        plt.imshow(epi_mask[:, :, slice_idx].T, cmap=epi_cmap, origin='lower', vmin=0, vmax=2)
        cb = plt.colorbar(ticks=[0, 1, 2])
        cb.ax.set_yticklabels(['0 (off)', '1x', '2x'])
        plt.title('EPI Mask')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(mask_dir / 'epi_mask.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save lesion mask for reference
        fig_lesion = plt.figure(figsize=(6, 6))
        plt.imshow(lesion_mask[:, :, slice_idx].T, cmap='tab10', origin='lower')
        plt.title('Lesion Labels')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(mask_dir / 'lesion_labels.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Save eroded mask for reference
        fig_eroded = plt.figure(figsize=(6, 6))
        plt.imshow(eroded_mask[:, :, slice_idx].T, cmap='tab10', origin='lower')
        plt.title('Eroded Lesion Labels')
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(mask_dir / 'eroded_labels.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # Row 3: Masked results
    axes[2, 0].imshow(t1_masked[:, :, slice_idx].T, cmap='gray', origin='lower', vmin=t1_vmin, vmax=t1_vmax)
    axes[2, 0].set_title('T1 Masked', fontsize=14, fontweight='bold', color='blue')
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(flair_masked[:, :, slice_idx].T, cmap='gray', origin='lower', vmin=flair_vmin, vmax=flair_vmax)
    axes[2, 1].set_title('FLAIR Masked', fontsize=14, fontweight='bold', color='blue')
    axes[2, 1].axis('off')
    
    axes[2, 2].imshow(phase_masked[:, :, slice_idx].T, cmap='gray', origin='lower', vmin=phase_vmin, vmax=phase_vmax)
    axes[2, 2].set_title('Phase Masked', fontsize=14, fontweight='bold', color='blue')
    axes[2, 2].axis('off')
    
    axes[2, 3].imshow(epi_masked[:, :, slice_idx].T, cmap='gray', origin='lower', vmin=epi_vmin, vmax=epi_masked_vmax)
    axes[2, 3].set_title('EPI Masked', fontsize=14, fontweight='bold', color='blue')
    axes[2, 3].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
        log_file = Path(save_path).parent / 'masking_debug.log'
        print(f"Debug log: {log_file}")
    else:
        plt.show()
        print(f"Debug log: masking_debug.log")
    
    plt.close()


def main(subject_dir, output_dir='./presentation_figures', random_seed=None, 
         save_individual=True, prefer_multiple=False,
         require_prl=False, require_cvs=False, from_alpaca_output=False,
         interactive=False):
    """Generate all visualization figures"""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("Loading subject data...")
    t1, flair, epi, phase, labeled, eroded = load_subject_data(subject_dir, from_alpaca_output)
    
    # Load predictions if filtering by PRL/CVS or in interactive mode
    predictions_df = None
    probabilities_df = None
    allowed_lesion_ids = None
    
    if require_prl or require_cvs or interactive:
        if from_alpaca_output or interactive:
            print("Loading ALPaCA predictions...")
            predictions_df, probabilities_df = load_predictions(subject_dir)
            
            if predictions_df is None and (require_prl or require_cvs):
                print("ERROR: No predictions found! Cannot filter by PRL/CVS.")
                print(f"Expected predictions.csv in: {subject_dir}/predictions/")
                return
    
    if require_prl or require_cvs:
        # Filter lesions by criteria
        allowed_lesion_ids, filtered_df = filter_lesions_by_criteria(
            predictions_df, require_prl, require_cvs
        )
        
        criteria_str = []
        if require_prl:
            criteria_str.append("PRL")
        if require_cvs:
            criteria_str.append("CVS")
        
        print(f"  Found {len(allowed_lesion_ids)} lesions matching criteria: {' + '.join(criteria_str)}")
        
        if len(allowed_lesion_ids) == 0:
            print("  No lesions match the specified criteria!")
            return
    
    # INTERACTIVE MODE
    if interactive:
        interactive_browse(t1, flair, epi, phase, labeled, eroded,
                          output_dir, predictions_df, probabilities_df,
                          allowed_lesion_ids)
        return
    
    # NORMAL MODE (rest of the function unchanged)
    print("Finding good lesion to visualize...")
    lesion_id, lesion_center = find_good_lesion(
        labeled, 
        random_seed=random_seed, 
        prefer_multiple=prefer_multiple,
        allowed_lesion_ids=allowed_lesion_ids
    )
    
    if lesion_id is None:
        print("No suitable lesion found!")
        return
    
    print(f"Using lesion {lesion_id} at center {lesion_center}")
    
    # Adjust center for multi-lesion to show both lesions better
    if prefer_multiple:
        adjusted_center = adjust_center_for_multi_lesion(labeled, lesion_id, lesion_center)
        patch_center = adjusted_center
    else:
        patch_center = lesion_center
    
    # Figure 1: Full slice view
    print("\n[1/5] Creating full slice view...")
    visualize_full_slice(t1, flair, epi, phase, labeled, 
                        patch_center[2],
                        save_path=output_dir / 'fig1_full_slice.png',
                        save_individual=save_individual)
    
    # Extract patch using your existing code
    print("[2/5] Extracting patch...")
    PATCH_RADIUS = 12
    starts = patch_center - PATCH_RADIUS
    ends = patch_center + PATCH_RADIUS - 1
    
    patch_tensor = extract_patch(
        lesion_id, starts, ends,
        t1, flair, epi, phase,
        labeled, eroded,
        rotate_patches=False
    )
    
    # Figure 2: 3D patch visualization
    print("[3/5] Creating patch visualization...")
    visualize_patch_3d(patch_tensor, 
                      save_path=output_dir / 'fig2_patch_3d.png',
                      save_individual=save_individual)
    
    # Figure 3: Attention masks
    print("[4/5] Creating attention mask visualization...")
    visualize_attention_mask(labeled, eroded, lesion_id, starts, ends,
                            save_path=output_dir / 'fig3_attention_masks.png')
    
    # Figure 4: Before/after masking
    print("[5/5] Creating masking effect visualization...")
    visualize_masked_comparison(t1, flair, epi, phase, labeled, eroded,
                               lesion_id, starts, ends,
                               save_path=output_dir / 'fig4_masking_effect.png')
    
    # Figure 5: Annotated with predictions (if available)
    if predictions_df is not None:
        print("[6/6] Creating annotated prediction visualization...")
        visualize_predictions_annotated(t1, flair, epi, phase, labeled,
                                       lesion_id, starts, ends,
                                       predictions_df, probabilities_df,
                                       save_path=output_dir / 'fig5_predictions_annotated.png')
    
    print(f"\n✓ All figures saved to {output_dir}")
    if save_individual:
        print(f"✓ Individual images saved to {output_dir / 'individual'}")
    print("\nGenerated figures:")
    print("  fig1_full_slice.png - All modalities at same slice")
    print("  fig2_patch_3d.png - 24x24x24 patch (center slice)")
    print("  fig3_attention_masks.png - Attention masks")
    print("  fig4_masking_effect.png - Raw → Mask → Masked (accurate!)")
    if predictions_df is not None:
        print("  fig5_predictions_annotated.png - With prediction labels (for inspection)")
    
    if random_seed is not None:
        print(f"\nUsed random seed: {random_seed}")
        print("Run with different --seed values to see different lesions!")
    if prefer_multiple:
        print("\nSearched for patches with multiple lesions to demonstrate attention masking!")
    if require_prl or require_cvs:
        criteria_str = []
        if require_prl:
            criteria_str.append("PRL")
        if require_cvs:
            criteria_str.append("CVS")
        print(f"\nFiltered to show: {' + '.join(criteria_str)}")
    
    print(f"\n📝 Debug log: {output_dir / 'masking_debug.log'}")
    print("   (Contains detailed mask analysis for troubleshooting)")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate ALPaCA visualization figures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with preprocessed images
  python visualize_patches.py /path/to/preprocessed/

  # Use ALPaCA output directory (with predictions)
  python visualize_patches.py /path/to/results/test/ --alpaca-output
  
  # INTERACTIVE MODE - Browse and save good examples!
  python visualize_patches.py /path/to/results/test/ --alpaca-output --interactive
  python visualize_patches.py /path/to/results/test/ --alpaca-output --interactive --prl
  
  # Find a PRL
  python visualize_patches.py /path/to/results/test/ --alpaca-output --prl
  
  # Find a CVS
  python visualize_patches.py /path/to/results/test/ --alpaca-output --cvs
  
  # Find a lesion that is BOTH PRL and CVS (rare!)
  python visualize_patches.py /path/to/results/test/ --alpaca-output --prl --cvs
  
  # Find a PRL with other lesions nearby
  python visualize_patches.py /path/to/results/test/ --alpaca-output --prl --multi-lesion
  
  # Try different random PRLs
  python visualize_patches.py /path/to/results/test/ --alpaca-output --prl --seed 42
        """
    )
    
    parser.add_argument('subject_dir', 
                       help='Path to preprocessed images or ALPaCA output directory')
    parser.add_argument('--output-dir', default='./presentation_figures',
                       help='Output directory for figures (default: ./presentation_figures)')
    parser.add_argument('--alpaca-root', 
                       help='Path to ALPaCA_python directory (if not auto-detected)')
    parser.add_argument('--alpaca-output', action='store_true',
                       help='Input is ALPaCA output directory (with predictions/ and preprocessed/ subfolders)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for lesion selection')
    parser.add_argument('--multi-lesion', action='store_true',
                       help='Prefer patches with multiple lesions')
    parser.add_argument('--prl', action='store_true',
                       help='Only show predicted PRLs (requires --alpaca-output)')
    parser.add_argument('--cvs', action='store_true',
                       help='Only show predicted CVS (requires --alpaca-output)')
    parser.add_argument('--no-individual', action='store_true',
                       help='Do not save individual images')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive browsing mode (use h/j/k/l to move, n/p for next/prev, s to save, q to quit)')
    
    args = parser.parse_args()
    
    # Validate arguments
    if (args.prl or args.cvs) and not args.alpaca_output:
        parser.error("--prl and --cvs require --alpaca-output")
    
    # If user provided alpaca root, add it to path
    if args.alpaca_root:
        sys.path.insert(0, args.alpaca_root)
        print(f"Using alpaca package from: {args.alpaca_root}")
    
    main(args.subject_dir, args.output_dir, 
         random_seed=args.seed, 
         save_individual=not args.no_individual,
         prefer_multiple=args.multi_lesion,
         require_prl=args.prl,
         require_cvs=args.cvs,
         from_alpaca_output=args.alpaca_output,
         interactive=args.interactive)