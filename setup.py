from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="superpacman",
    version="0.1.0",
    author="Duane Nielsen",
    author_email="duanenielsen@example.com",
    description="The SuperPacman reinforcement learning environment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/duanenielsen/superpacman",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.11',
    install_requires=["torch", "torchrl", "torchvision", "matplotlib", "tqdm", "av"],
    extras_require={
        "dev": ['twine', 'wheel'],
    },
    entry_points={
        'console_scripts':
            'superpacman = superpacman.commands:main'
    }
)
