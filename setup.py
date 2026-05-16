from setuptools import setup, find_packages
from setuptools.command.install import install
import subprocess, sys, os, json

KAGGLE_USERNAME = "rayhan313540001"


def _setup_kaggle_auth():
    """Ensure ~/.kaggle/kaggle.json exists, converting access_token if needed."""
    kaggle_dir = os.path.expanduser("~/.kaggle")
    kaggle_json = os.path.join(kaggle_dir, "kaggle.json")
    access_token = os.path.join(kaggle_dir, "access_token")

    if os.path.exists(kaggle_json):
        return kaggle_json

    if os.path.exists(access_token):
        with open(access_token) as f:
            token = f.read().strip()
        os.makedirs(kaggle_dir, exist_ok=True)
        with open(kaggle_json, "w") as f:
            json.dump({"username": KAGGLE_USERNAME, "key": token}, f)
        os.chmod(kaggle_json, 0o600)
        print("[*] Converted ~/.kaggle/access_token -> ~/.kaggle/kaggle.json")
        return kaggle_json

    return None


_DOWNLOAD_SCRIPT = """
import subprocess, sys
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'kagglehub', 'kagglesdk'], capture_output=True)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kagglehub', '-q'])
import kagglehub
kagglehub.competition_download(__COMPETITION__, path="__DATADIR__")
print("Done")
"""


def _download_competition(competition_handle, data_dir):
    script = _DOWNLOAD_SCRIPT.replace("__COMPETITION__", repr(competition_handle))
    script = script.replace("__DATADIR__", repr(data_dir))
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True,
        env={**os.environ, "KAGGLE_USERNAME": KAGGLE_USERNAME}
    )
    if result.returncode != 0 or "Done" not in result.stdout:
        print(f"    kagglehub error:\n{result.stderr.strip()}")
        print(f"    stdout: {result.stdout.strip()}")
        return False
    return True


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

        if _download_competition("nycu-data-mining-assignment-3", "data"):
            print("[*] Downloaded successfully.")
            return

        print("[!] Automatic download failed — invite-only competition.")
        print("    1. Accept invite and download manually:")
        print("       https://www.kaggle.com/t/7177902eb8b34b25a75e932d4e235b32")
        print("    2. Place train/ and test/ folders + sample_submission.csv in data/")


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
