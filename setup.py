"""Setup script for Multilingual Safety Benchmark (MSB)"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="msb-complete",
    version="1.0.0",
    author="MSB Contributors",
    author_email="msb@example.com",
    description="A comprehensive framework for evaluating LLM safety across languages",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/msb-complete",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "msb=msb.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "msb": ["data/*.json", "configs/*.yaml"],
    },
)