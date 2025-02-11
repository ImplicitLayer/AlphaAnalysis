from setuptools import setup, find_packages

setup(
    name="AlphaAnalysis",
    version="0.1.0",
    author="ArtemBurenok",
    author_email="burenok023@gmail.com",
    description="Library for analyzing financial data using ML and classical approaches",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="Apache License, version 2.0",
    url="https://github.com/ImplicitLayer/AlphaAnalysis",  # Замените на ваш репозиторий
    packages=find_packages(),
    install_requires=[
        "pandas",
        "yfinance",
        "alpha_vantage",
        "requests",
        "matplotlib",
        "seaborn",
        "numpy",
        "statsmodels",
        "scikit-learn",
        "arch",
        "xgboost",
        "catboost",
        "tensorflow",
        "nltk",
        "textblob"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.10",
)
