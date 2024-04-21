from setuptools import setup, find_packages
import subprocess

import re

VERSION_PATTERN = r"""
    v?
    (?:
        (?:(?P<epoch>[0-9]+)!)?                           # epoch
        (?P<release>[0-9]+(?:\.[0-9]+)*)                  # release segment
        (?P<pre>                                          # pre-release
            [-_\.]?
            (?P<pre_l>(a|b|c|rc|alpha|beta|pre|preview))
            [-_\.]?
            (?P<pre_n>[0-9]+)?
        )?
        (?P<post>                                         # post release
            (?:-(?P<post_n1>[0-9]+))
            |
            (?:
                [-_\.]?
                (?P<post_l>post|rev|r)
                [-_\.]?
                (?P<post_n2>[0-9]+)?
            )
        )?
        (?P<dev>                                          # dev release
            [-_\.]?
            (?P<dev_l>dev)
            [-_\.]?
            (?P<dev_n>[0-9]+)?
        )?
    )
    (?:\+(?P<local>[a-z0-9]+(?:[-_\.][a-z0-9]+)*))?       # local version
"""

def is_valid_pep440_version(version_string):
    _regex = re.compile(r"^\s*" + VERSION_PATTERN + r"\s*$", re.VERBOSE | re.IGNORECASE)
    return bool(_regex.search(version_string))

# search the tags for the most recent python package verions number
TAGS = subprocess.run(['git', 'tag', '--sort=creatordate', '--merged'], stdout=subprocess.PIPE).stdout.decode("utf-8").split('\n')
VERSION = 'invalid'
for tag in reversed(TAGS):
    if is_valid_pep440_version(tag):
        VERSION = tag
        break
print(f"found PACKAGE VERSION: {VERSION}")

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="superpacman",
    version=VERSION,
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
    install_requires=["torch", "torchrl", "torchvision", "matplotlib", "tqdm", "av", "moviepy", "hrid", "PyQt5", "kornia"],
    extras_require={
        "dev": ['twine', 'wheel', 'setuptools', 'sphinx', 'sphinx_rtd_theme', 'ghp_import', 'sphinx-argparse', 'packaging'],
    },
    entry_points={
        'console_scripts':
            'superpacman = superpacman.commands:main'
    }
)
