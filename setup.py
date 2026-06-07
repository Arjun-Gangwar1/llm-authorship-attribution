# setup.py
from setuptools import setup, find_packages

setup(
    name="llm-text-classifier",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    author="Arjun Gangwar",
    description="12-class LLM-generated text classifier",
    python_requires=">=3.10",
)