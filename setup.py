from setuptools import setup, find_packages
import subprocess
import os

superpacman_remote_version = subprocess.run(['git', 'describe', '--tags'], stdout=subprocess.PIPE).stdout.decode("utf-8").strip()
assert "." in superpacman_remote_version

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="superpacman",
    version=superpacman_remote_version,
    author="Duane Nielsen",
    author_email="duanenielsen@example.com",
    description="The SuperPacman reinforcement learning environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/duanenielsen/superpacman",
    packages=find_packages(),
    package_data={
        'superpacman': ['checkpoints/*']
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.8',
    install_requires=["torch", "torchrl", "torchvision", "matplotlib", "tqdm", "av", "moviepy"],
    extras_require={
        "dev": ['twine', 'wheel'],
    },
    entry_points={
        'console_scripts':
            'superpacman = superpacman.commands:main'
    }
)
