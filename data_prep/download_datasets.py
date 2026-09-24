"""
Automated Downloader and Extractor for VisDrone2019-DET and Seraphim Drone Datasets.

Features:
- Downloads and unzips VisDrone2019-DET train/val splits.
- Downloads Seraphim Drone Detection Dataset from HuggingFace Hub.
- Validates downloaded file integrity and creates organized directory structures.
"""

import argparse
import os
import sys
import ssl
import zipfile
from pathlib import Path
import certifi
import requests
from tqdm import tqdm

from config import RAW_DIR

# Fix Windows console encoding if needed
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Configure SSL context using certifi
ssl_context = ssl.create_default_context(cafile=certifi.where())

# URLs / identifiers for datasets
VISDRONE_URLS = {
    "train": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-train.zip",
    "val": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-val.zip",
    "test-dev": "https://github.com/ultralytics/assets/releases/download/v0.0.0/VisDrone2019-DET-test-dev.zip"
}

# Verified HuggingFace Seraphim dataset repository ID
DEFAULT_SERAPHIM_REPO = "lgrzybowski/seraphim-drone-detection-dataset"


def download_url(url: str, output_path: Path):
    """Download a file with requests streaming and a progress bar."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[*] Downloading {url} -> {output_path.name}...")
    
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total_size = int(response.headers.get('content-length', 0))

    with open(output_path, "wb") as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)
                bar.update(len(chunk))


def extract_zip(zip_path: Path, extract_to: Path):
    """Safely extract a ZIP archive."""
    print(f"[*] Extracting {zip_path.name} to {extract_to}...")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"[✓] Extracted {zip_path.name}")


def setup_visdrone(dest_dir: Path, download_test: bool = False):
    """Download and prepare raw VisDrone2019-DET dataset."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    splits = ["train", "val"]
    if download_test:
        splits.append("test-dev")

    print("\n" + "=" * 60)
    print("STEP 1: VISDRONE2019-DET DATASET SETUP")
    print("=" * 60)

    for split in splits:
        split_dir = dest_dir / f"VisDrone2019-DET-{split}"
        if split_dir.exists() and any(split_dir.iterdir()):
            print(f"[✓] VisDrone {split} already extracted at {split_dir}")
            continue

        zip_file = dest_dir / f"VisDrone2019-DET-{split}.zip"
        if not zip_file.exists():
            url = VISDRONE_URLS.get(split)
            if not url:
                print(f"[!] Warning: No URL registered for split '{split}'")
                continue
            try:
                download_url(url, zip_file)
            except Exception as e:
                print(f"[!] Direct download failed ({e}).")
                print(f"    Please manually download {zip_file.name} and place it in {dest_dir}")
                continue

        if zip_file.exists():
            extract_zip(zip_file, dest_dir)


def setup_seraphim(dest_dir: Path, repo_id: str = DEFAULT_SERAPHIM_REPO):
    """
    Download Seraphim Drone Detection Dataset from HuggingFace.
    Requires `huggingface_hub` package.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    print("\n" + "=" * 60)
    print("STEP 2: SERAPHIM DRONE DATASET SETUP (HuggingFace)")
    print("=" * 60)

    if any(dest_dir.glob("**/*.jpg")) or any(dest_dir.glob("**/*.png")):
        print(f"[✓] Seraphim dataset images already found at {dest_dir}")
        return

    try:
        from huggingface_hub import snapshot_download
        print(f"[*] Downloading dataset repo '{repo_id}' from HuggingFace Hub...")
        snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=str(dest_dir),
            local_dir_use_symlinks=False,
            resume_download=True
        )
        print(f"[✓] Seraphim dataset successfully downloaded to {dest_dir}")
    except ImportError:
        print("[!] Error: `huggingface_hub` is required. Run: pip install huggingface_hub")
    except Exception as e:
        print(f"[!] HuggingFace download exception: {e}")
        print(f"    You can also manually clone or copy the Seraphim dataset folder into {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description="Download and extract aerial datasets (VisDrone & Seraphim).")
    parser.add_argument("--visdrone-dir", type=str, default=str(RAW_DIR / "visdrone"),
                        help="Destination directory for raw VisDrone data")
    parser.add_argument("--seraphim-dir", type=str, default=str(RAW_DIR / "seraphim"),
                        help="Destination directory for raw Seraphim data")
    parser.add_argument("--seraphim-repo", type=str, default=DEFAULT_SERAPHIM_REPO,
                        help="HuggingFace dataset repository ID for Seraphim")
    parser.add_argument("--include-test", "--full", action="store_true", default=True,
                        help="Download all splits including VisDrone test-dev (default: True)")
    parser.add_argument("--skip-visdrone", action="store_true", help="Skip VisDrone download")
    parser.add_argument("--skip-seraphim", action="store_true", help="Skip Seraphim download")

    args = parser.parse_args()

    if not args.skip_visdrone:
        setup_visdrone(Path(args.visdrone_dir), download_test=args.include_test)

    if not args.skip_seraphim:
        setup_seraphim(Path(args.seraphim_dir), repo_id=args.seraphim_repo)

    print("\n[✓] Full data download process finished.")


if __name__ == "__main__":
    main()
