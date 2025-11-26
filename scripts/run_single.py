#!/usr/bin/env python3
"""
Run ALPaCA on a single subject.

Two modes:

1. Explicit paths:
    python scripts/run_single.py \
        --t1 /path/to/t1.nii.gz \
        --flair /path/to/flair.nii.gz \
        --epi /path/to/epi_mag.nii.gz \
        --phase /path/to/epi_phase.nii.gz \
        --labels /path/to/lesion_labels.nii.gz \
        --output /path/to/output

2. Auto-detect:
    python scripts/run_single.py \
        --subject-dir /path/to/subject/session_date \
        --output /path/to/output
        
    Looks for files matching these patterns:
    - *_T1_MTTE.nii.gz or T1_MTTE.nii.gz or t1.nii.gz
    - *_FL_MTTE.nii.gz or FLAIR_MTTE.nii.gz or flair.nii.gz
    - *_T2star_mag_MTTE.nii.gz or epi_mag.nii.gz
    - *_T2star_phase_unwrapped_MTTE.nii.gz or epi_phase.nii.gz
    - Lesion_Index*.nii.gz or lesion_labels.nii.gz
"""

import argparse
import sys
from pathlib import Path

# Suppress NVML initialization warning from PyTorch
import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='Can\'t initialize NVML')

sys.path.insert(0, str(Path(__file__).parent.parent))


from alpaca.preprocessing.minimal_preprocess import minimal_preprocess
from alpaca.models.make_predictions import make_predictions


def find_file(directory, patterns):
    """Find first file matching any of the patterns."""
    directory = Path(directory)
    for pattern in patterns:
        matches = list(directory.glob(pattern))
        if matches:
            return str(matches[0])
    return None


def auto_detect_files(subject_dir):
    """Auto-detect files from subject directory."""
    subject_dir = Path(subject_dir)
    
    # Define search patterns (in order of preference)
    patterns = {
        't1': ['*_T1_MTTE.nii.gz', 'T1_MTTE.nii.gz', 't1.nii.gz', 'T1.nii.gz', 't1_final.nii.gz'],
        'flair': ['*_FL_MTTE.nii.gz', 'FLAIR_MTTE.nii.gz', 'flair.nii.gz', 'FLAIR.nii.gz', 'flair_final.nii.gz'],
        'epi': ['*_T2star_mag_MTTE.nii.gz', '*_T2star_MTTE.nii.gz', 
                'epi_mag.nii.gz', 'EPI_mag.nii.gz', 't2star_mag.nii.gz', 'epi_final.nii.gz'],
        'phase': ['*_T2star_phase_unwrapped_MTTE.nii.gz', 
                  'epi_phase_unwrapped.nii.gz', 'epi_phase.nii.gz',
                  't2star_phase_unwrapped.nii.gz', 'phase.nii.gz', 'phase_final.nii.gz'],
        'labels': ['Lesion_Index_spectral.nii.nii.gz', 'Lesion_Index_Spectral.nii.nii.gz',
                   'Lesion_Index_spectral.nii.gz', 'lesion_labels.nii.gz', 
                   'lesion_mask.nii.gz', 'labeled_candidates.nii.gz'],
        'eroded': ['eroded_candidates.nii.gz', 'eroded_labels.nii.gz', 'eroded_candidates.nii.gz']
    }
    
    files = {}
    for key, pattern_list in patterns.items():
        found = find_file(subject_dir, pattern_list)
        if found:
            files[key] = found
    
    return files


def main():
    parser = argparse.ArgumentParser(
        description='Run ALPaCA on a single subject',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Mode selection
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--subject-dir', 
                           help='Auto-detect files from subject directory')
    mode_group.add_argument('--t1',
                           help='T1-weighted image (explicit mode)')
    
    # Explicit mode arguments (only needed if not using --subject-dir)
    parser.add_argument('--flair', help='FLAIR image')
    parser.add_argument('--epi', help='EPI magnitude image')
    parser.add_argument('--phase', help='EPI phase image (unwrapped)')
    parser.add_argument('--labels', help='Labeled lesion candidates')
    parser.add_argument('--eroded-labels', help='Pre-eroded labels (optional)')

    
    # Required for both modes
    parser.add_argument('--model-dir', default=None, 
                       help='Directory with model weights (default ALPaCA_python/models)')
    parser.add_argument('--output', required=True,
                       help='Output directory')
    parser.add_argument('--skip-erosion', action='store_true')

    # Inference options
    parser.add_argument('--n-patches', type=int, default=20,
                       help='Number of patches per lesion (default: 20)')
    parser.add_argument('--n-models', type=int, default=10,
                       help='Number of CV models to use (default: 10)')
    parser.add_argument('--no-rotate', action='store_true',
                       help='Disable random patch rotation')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--return-prob-maps', action='store_true',
                       help='Save probability maps')
    
    # Thresholds
    parser.add_argument('--lesion-threshold', default='youdens_j',
                       choices=['youdens_j', 'specificity', 'sensitivity'])
    parser.add_argument('--prl-threshold', default='youdens_j',
                       choices=['youdens_j', 'specificity', 'sensitivity'])
    parser.add_argument('--cvs-threshold', default='youdens_j',
                       choices=['youdens_j', 'specificity', 'sensitivity'])
    
    parser.add_argument('--verbose', action='store_true')
    
    args = parser.parse_args()
    
    verbose = args.verbose
    
    if verbose:
        print("\n╔" + "═"*78 + "╗")
        print("║" + " ALPaCA Single Subject Pipeline ".center(78) + "║")
        print("╚" + "═"*78 + "╝")
    
    # Determine mode and get file paths
    if args.subject_dir:
        # Auto-detect mode
        if verbose:
            print(f"\nAuto-detecting files from: {args.subject_dir}")
        
        files = auto_detect_files(args.subject_dir)
        
        required = ['t1', 'flair', 'epi', 'phase', 'labels']
        missing = [k for k in required if k not in files]
        
        if missing:
            print(f"Error: Could not auto-detect required files: {missing}")
            print(f"\nSearched in: {args.subject_dir}")
            print("Consider using explicit mode with --t1, --flair, etc.")
            sys.exit(1)
        
        if verbose:
            for key, path in files.items():
                if key in required and key or key == 'eroded':
                    print(f"  {key:8s}: {Path(path).name}")
        
        t1_path = files['t1']
        flair_path = files['flair']
        epi_path = files['epi']
        phase_path = files['phase']
        labels_path = files['labels']
        eroded_path = files.get('eroded', None)
        
    else:
        # Explicit mode
        required_args = ['t1', 'flair', 'epi', 'phase', 'labels']
        missing = [arg for arg in required_args if getattr(args, arg) is None]
        
        if missing:
            print(f"Error: Missing required arguments: {missing}")
            print("In explicit mode, you must provide: --t1, --flair, --epi, --phase, --labels")
            sys.exit(1)
        
        t1_path = args.t1
        flair_path = args.flair
        epi_path = args.epi
        phase_path = args.phase
        labels_path = args.labels
        eroded_path = args.eroded_labels
    
    # Validate all paths exist
    for name, path in [('t1', t1_path), ('flair', flair_path), 
                       ('epi', epi_path), ('phase', phase_path), 
                       ('labels', labels_path)]:
        if not Path(path).exists():
            print(f"Error: {name} file not found: {path}")
            sys.exit(1)
    
    if eroded_path and not Path(eroded_path).exists():
        print(f"Error: eroded-labels file not found: {eroded_path}")
        sys.exit(1)
    
    if args.model_dir and not Path(args.model_dir).exists():
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    output_dir = Path(args.output)
    
    # Run pipeline
    
    preprocessed = minimal_preprocess(
        t1_path=t1_path,
        flair_path=flair_path,
        epi_path=epi_path,
        phase_path=phase_path,
        labels_path=labels_path,
        eroded_candidates_path=eroded_path,
        skip_erosion=args.skip_erosion,
        output_dir=str(output_dir / "preprocessed"),
        verbose=verbose
    )
    
    results = make_predictions(
        t1=preprocessed['t1'],
        flair=preprocessed['flair'],
        epi=preprocessed['epi'],
        phase=preprocessed['phase'],
        labeled_candidates=preprocessed['labeled_candidates'],
        eroded_candidates=preprocessed['eroded_candidates'],
        model_dir=args.model_dir,
        output_dir=str(output_dir / "predictions"),
        lesion_priority=args.lesion_threshold,
        prl_priority=args.prl_threshold,
        cvs_priority=args.cvs_threshold,
        n_patches=args.n_patches,
        n_models=args.n_models,
        rotate_patches=not args.no_rotate,
        return_probabilities=args.return_prob_maps,
        random_seed=args.seed,
        verbose=verbose
    )
    
if __name__ == "__main__":
    main()