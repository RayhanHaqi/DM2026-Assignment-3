from setuptools import setup, find_packages
from setuptools import Command
import subprocess, sys, os

KAGGLE_URL = "https://www.kaggle.com/competitions/<COMPETITION_NAME>/data"


class DataCommand(Command):
    """Download Kaggle competition data using kagglehub."""
    description = "download Kaggle competition data (requires ~/.kaggle/kaggle.json)"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            import kagglehub
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
            import kagglehub

        os.makedirs("data", exist_ok=True)

        # Check if data already exists
        if os.path.isdir("data/train") or os.path.isdir("data/train/train"):
            print("[*] Data already exists in data/. Skipping download.")
            return

        print("[*] Attempting Kaggle download...")
        print("[!] Requires: ~/.kaggle/kaggle.json (Kaggle API key)")
        print("[!] If this fails, download manually from the Kaggle competition page.")

        for handle in [
            "dm-2026-assignment-3",
            "human-activity-recognition-spring-2026",
        ]:
            try:
                print(f"    Trying: {handle}")
                kagglehub.competition_download(handle, path="data")
                print(f"[*] Downloaded: {handle}")
                return
            except Exception:
                continue

        print("[!] Automatic download failed.")
        print("    Download data manually from the Kaggle competition page.")
        print("    Place train/ and test/ folders + sample_submission.csv in data/")


setup(
    name="dm2026_asg3",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "xgboost",
        "lightgbm",
        "optuna",
        "matplotlib",
        "seaborn",
    ],
    cmdclass={
        "data": DataCommand,
    },
)
