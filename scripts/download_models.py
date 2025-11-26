#!/usr/bin/env python3
"""
Download ALPaCA models from Zenodo and extract .pt files.
Usage: python download_models.py [output_dir]
"""

import sys
import zipfile
from pathlib import Path
from urllib.request import urlretrieve
from tqdm import tqdm

ZENODO_URL = "https://zenodo.org/records/17215591/files/ALPaCA%20v1.0.0.zip?download=1"
MODEL_PATH_IN_ZIP = "ALPaCA 2/inst/extdata/"

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_and_extract(output_dir=None):
    if output_dir is None:
        # Default: ../models relative to this script
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "models"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = output_dir / "alpaca_temp.zip"
    
    # Download
    print(f"Downloading from Zenodo (2.3 GB)...")
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc='ALPaCA v1.0.0.zip') as t:
        urlretrieve(ZENODO_URL, zip_path, reporthook=t.update_to)
    
    # Extract only .pt files
    print(f"Extracting models...")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        for file in zf.namelist():
            if file.startswith(MODEL_PATH_IN_ZIP) and file.endswith('.pt'):
                filename = Path(file).name
                print(f"  Extracting {filename}")
                data = zf.read(file)
                (output_dir / filename).write_bytes(data)
    
    # Cleanup
    zip_path.unlink()
    print(f"\nDone. Models in: {output_dir.absolute()}")

if __name__ == "__main__":
    output_dir = sys.argv[1] if len(sys.argv) > 1 else None
    download_and_extract(output_dir)