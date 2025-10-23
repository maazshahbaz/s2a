"""
Setup configuration for S2A Python SDK
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')

# Read requirements
requirements = []
try:
    with open('requirements.txt') as f:
        requirements = [line.strip() for line in f.read().splitlines() 
                       if line.strip() and not line.strip().startswith('#')]
except FileNotFoundError:
    # Fallback requirements if file not found
    requirements = [
        "httpx>=0.25.0",
        "pydantic>=2.0.0", 
        "typing-extensions>=4.0.0"
    ]

setup(
    name="s2a-sdk",
    version="1.0.4",
    author="99Technologies AI",
    author_email="support@99technologies.ai",
    description="Official Python SDK for S2A Speech-to-Actions Platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/99technologies-ai/s2a-sdk-python",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Multimedia :: Sound/Audio :: Speech",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "audio": ["librosa>=0.10.0", "soundfile>=0.12.0"],
        "dev": ["pytest>=7.0.0", "pytest-asyncio>=0.21.0", "black", "flake8"],
        "all": ["librosa>=0.10.0", "soundfile>=0.12.0", "pytest>=7.0.0", "pytest-asyncio>=0.21.0", "black", "flake8"]
    },
    keywords="speech-to-text transcription ai business-intelligence s2a audio nlp",
    project_urls={
        "Bug Reports": "https://github.com/99technologies-ai/s2a/issues",
        "Documentation": "https://docs.bytepulseai.com/sdk/python",
        "Source": "https://github.com/99technologies-ai/s2a-sdk-python",
        "API Reference": "https://api.bytepulseai.com/docs"
    },
)