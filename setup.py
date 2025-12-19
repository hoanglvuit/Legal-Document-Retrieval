"""Setup script for legal_retrieval package"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="legal-retrieval",
    version="1.0.0",
    author="Legal Document Retrieval Team",
    description="A two-stage legal document retrieval system using Bi-Encoder and Cross-Encoder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/Legal-Document-Retrieval",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "legal-retrieval-train-bi=scripts.train_bi:main",
            "legal-retrieval-train-cross=scripts.train_cross:main",
            "legal-retrieval-predict-bi=scripts.predict_bi:main",
            "legal-retrieval-predict-cross=scripts.predict_cross:main",
        ],
    },
)
