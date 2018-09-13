import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()

setuptools.setup(
	name="neuralocalize",
	version="0.0.1",
	author="Ido Kessler Noam Loya Itay Itzhak",
	author_email="idokessler@mail.tau.ac.il",
	description="Task-free MRI predicts individual differences in brain activity during task performence.",
	long_description=long_description,
	long_description_content_type="text/markdown",
	url="https://github.com/kessido/Neuroscience-seminar",
	packages=setuptools.find_packages(),
	entry_points = {'console_scripts': ['localize = neuralocalize.localize:main']},
	classifiers=[
		"Programming Language :: Python :: 3",
		"License :: OSI Approved :: MIT License",
		"Operating System :: OS Independent",
	],include_package_data=True
)