from setuptools import setup, find_packages

setup(
    name="alpha_analysis",
    version="1.2.2",
    author="ArtemBurenok",
    author_email="burenok023@gmail.com",
    description="Library for analyzing financial data_preprocessing using ML and classical approaches",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache License, version 2.0",
    url="https://github.com/ImplicitLayer/AlphaAnalysis",
    packages=find_packages(),
    install_requires=[
        "pandas",
        "matplotlib",
        "numpy",
        "statsmodels",
        "scikit-learn",
        "arch",
        "xgboost",
        "nltk",
        "textblob",
        "hmmlearn",
        "shap",
        "pywt",
        "torch"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10",
)
