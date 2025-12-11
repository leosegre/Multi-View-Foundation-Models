import os
import subprocess
import sys

# Try to import huggingface_hub, install if missing
try:
    from huggingface_hub import snapshot_download
except ImportError:
    print("📦 Installing huggingface_hub...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub"])
    from huggingface_hub import snapshot_download


def run_demo():
    # --- Configuration ---
    MODEL_REPO = "Leoseg/dinov2_reg"
    DATA_REPO = "Leoseg/pikachu"

    # Local folders
    EXP_ROOT_DIR = "experiments"
    DATA_ROOT_DIR = "demo_data"

    # The name of the experiment (folder name inside experiments/)
    EXP_NAME = "dinov2_reg"

    # --- Step 1: Download Model ---
    print(f"\n⬇️  Downloading Model: {MODEL_REPO}...")
    # This downloads into experiments/dinov2_reg/
    model_dir = os.path.join(EXP_ROOT_DIR)
    snapshot_download(
        repo_id=MODEL_REPO,
        local_dir=model_dir,
        local_dir_use_symlinks=False  # Download real files, not links
    )

    # --- Step 2: Download Data ---
    print(f"\n⬇️  Downloading Data: {DATA_REPO}...")
    # This downloads into demo_data/pikachu/
    # Note: We must specify repo_type="dataset" for dataset links
    scene_name = "pikachu"
    scene_dir = os.path.join(DATA_ROOT_DIR)
    snapshot_download(
        repo_id=DATA_REPO,
        repo_type="dataset",
        local_dir=scene_dir,
        local_dir_use_symlinks=False
    )

    # --- Step 3: Run the Command ---
    print("\n🚀 Starting the demo...")

    # Command construction
    # We point colmap_path to 'demo_data' because the script likely looks for {colmap_path}/{scene}
    cmd = [
        sys.executable, "test/correspondence.py",
        "--exp_directory", EXP_ROOT_DIR,
        "--exp_name", EXP_NAME,
        "--colmap_path", DATA_ROOT_DIR,
        "--fit3d",
        "--scene", scene_name,
        "--limit_corrs", "50"
    ]

    print(f"Executing: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running the demo: {e}")
    except KeyboardInterrupt:
        print("\n🛑 Demo stopped by user.")


if __name__ == "__main__":
    run_demo()