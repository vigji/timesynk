from setuptools import find_namespace_packages, setup

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

with open("requirements_dev.txt") as f:
    requirements_dev = f.read().splitlines()

with open("README.md") as f:
    long_description = f.read()

setup(
    name="timesynk",
    version="0.1.0",
    description="Time synchronisation utils to register different data flows and resample acquisitions.",
    install_requires=requirements,
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    extras_require=dict(dev=requirements_dev),
    packages=find_namespace_packages(where="src", exclude=("docs", "tests*")),
    package_dir={"": "src"},
    include_package_data=True,
    url="https://github.com/vigji/timesynk",
    author="Luigi Petrucco",
    author_email="luigi.petrucco@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows :: Windows 10",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Intended Audience :: Science/Research",
    ],
    zip_safe=False,
)
