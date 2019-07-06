import pathlib
from setuptools import setup, find_packages

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()

# This call to setup() does all the work
setup(
    name="signal-pipes",
    version="0.1.1",
    description="Stream operation for processing of (physiological) signals",
    long_description=README,
    long_description_content_type="text/markdown",
    url="",
    author="Department of Informatics, Faculty of Science, University of J.E. Purkyne in Usti nad Labem ",
    author_email="Jiri.Fiser@ujep.cz",
    license="Apache Software License",
    classifiers=[
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
    ],
    packages=find_packages(exclude=("tests",)),
    include_package_data=True,
    install_requires=["numpy", "scipy", 'h5py', 'matplotlib']
)
