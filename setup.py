from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess, sys, os


class InstallWithData(install):
    """Install package + dependencies, then download Kaggle data."""

    def run(self):
        install.run(self)
        self._download_data()

    def _download_data(self):
        try:
            import kagglehub
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "kagglehub"])
            import kagglehub

        os.makedirs("data", exist_ok=True)

        if os.path.isdir("data/train") or os.path.isdir("data/train/train"):
            print("[*] Data already exists in data/. Skipping download.")
            return

        print("[*] Downloading Kaggle competition data...")
        print("[!] Requires ~/.kaggle/kaggle.json (Kaggle API key)")

        for handle in [
            "dm-2026-assignment-3",
            "human-activity-recognition-spring-2026",
        ]:
            try:
                kagglehub.competition_download(handle, path="data")
                print(f"[*] Downloaded: {handle}")
                return
            except Exception:
                continue

        print("[!] Automatic download failed. Download manually from Kaggle.")
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
        "install": InstallWithData,
    },
)
