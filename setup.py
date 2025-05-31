"""Setup script for ai-lang."""

from setuptools import setup, find_packages

setup(
    name="ai-lang",
    version="0.1.0",
    description="A dependently-typed programming language implemented in Python",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "lark>=1.1.9",
        "rich>=13.7.1",
        "click>=8.1.7",
    ],
    extras_require={
        "dev": [
            "pytest>=8.1.1",
            "pytest-cov>=5.0.0",
            "mypy>=1.9.0",
            "black>=24.3.0",
            "ruff>=0.3.4",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai-lang=ai_lang.cli:main",
        ],
    },
)