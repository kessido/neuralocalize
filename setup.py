import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="neuralocalize",
    version="0.1dev",
    author="Ido Tavor",
    author_email="author@example.com",
    description="Task-free MRI predicts individual differences in brain activity during task performence.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/pypa/sampleproject",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

from distutils.core import setup

setup(
    name='neuralocalize',
    version='0.1dev',
    packages=['python'],  # TODO(loya) see if this name needs to be changed
    # TODO(loya) validate and rename. Might also need .py
    scripts=['neuralocalize'],
    long_description='''
    This package implements the model training and prediction described in Tavor el al,
    Task-free MRI predicts individual differences in brain activity during task performance:
    http://science.sciencemag.org/content/352/6282/216.
    ''',
    requires=['numpy', 'scikit-learn', 'nibabel', 'cifti'],
)
