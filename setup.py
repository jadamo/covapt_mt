from setuptools import setup, find_packages

setup(
    name="covapt_mt",
    version="0.1",
    author="Joe Adamo",
    author_email="jadamo@arizona.edu",
    description="multi-tracer version of covapt for doing SPHEREx inference",
    packages=find_packages(),
    python_requires='>=3.5,<3.9',
    install_requires=["build",
                      "numpy",
                      "scipy",
                      "nbodykit",
                      "pyyaml",
    ],
)