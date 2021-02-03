# setup.py

import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="devModel",
    version="0.0.1",
    author="Manuel Martinez",
    author_email="manuelmartinez27ale@gmail.com",
    description="An useful package to develop model and makes analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    url="https://github.com/Alemar1990/devModel",
    include_package_data=True,
    packages=setuptools.find_packages(),
    install_requires=[
        'tqdm==4.50.2',
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
