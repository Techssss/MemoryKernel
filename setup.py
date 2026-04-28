"""
MemoryKernel - Local-first project brain for AI agents
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    requirements = [
        line.strip()
        for line in requirements_file.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="memk",
    version="0.1.0",
    author="MemoryKernel Team",
    author_email="dev@memorykernel.dev",
    description="Project memory that AI agents can carry across sessions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Techssss/MemoryKernel",
    packages=find_packages(exclude=["tests", "benchmarks", "examples", "docs"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "memk=memk.cli.main:app",
            "memk-mcp=memk.mcp.server:main",
        ],
    },
    include_package_data=True,
    package_data={
        "memk": ["py.typed"],
    },
    keywords=[
        "memory",
        "ai",
        "agent",
        "knowledge-base",
        "vector-search",
        "local-first",
        "rag",
        "embeddings",
    ],
    project_urls={
        "Bug Reports": "https://github.com/Techssss/MemoryKernel/issues",
        "Source": "https://github.com/Techssss/MemoryKernel",
        "Documentation": "https://github.com/Techssss/MemoryKernel/tree/main/docs",
    },
)
