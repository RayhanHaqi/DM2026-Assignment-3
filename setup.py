from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess, sys, os, json


def _setup_kaggle_auth():
    """Ensure ~/.kaggle/kaggle.json exists, converting access_token if needed."""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    access_token = os.path.join(kaggle_dir, "access_token")

    if os.path.exists(kaggle_json):
        return True

    if os.path.exists(access_token):
        with open(access_token) as f:
            token = f.read().strip()
        # Token might be "KGAT_xxx" — just the key, no username prefix
        # Try key-only format that kagglehub accepts
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(kaggle_json, "w") as f:
            json.dump({"username": "token", "key": token}, f)
        os.chmod(kaggle_json, 0o600)
        print("[*] Converted ~/.kaggle/access_token -> ~/.kaggle/kaggle.json")
        return True

    return False


class InstallWithData(install):
    """Install package + dependencies, then download Kaggle data."""

    def run(self):
        install.run(self)
        self._download_data()

    def _download_data(self):
        os.makedirs("data", exist_ok=True)

        if os.path.isdir("data/train") or os.path.isdir("data/train/train"):
            print("[*] Data already exists in data/. Skipping download.")
            return

        _setup_kaggle_auth()

        print("[*] Downloading Kaggle competition data...")

        # Run download in subprocess to avoid kagglehub import version conflicts
        download_script = (
            "import kagglehub; "
            'kagglehub.competition_download("nycu-data-mining-assignment-3", path="data"); '
            'print("Done")'
        )
        result = subprocess.run(
            [sys.executable, "-c", download_script],
            capture_output=True, text=True
        )
        if result.returncode == 0 and "Done" in result.stdout:
            print("[*] Downloaded successfully.")
            return
        if result.stderr:
            print(f"    kagglehub error: {result.stderr.strip()[:300]}")

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
        "install": InstallWithData,
    },
)
