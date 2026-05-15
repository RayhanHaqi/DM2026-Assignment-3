from setuptools import setup, find_packages

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
)
