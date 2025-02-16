from setuptools import setup, find_packages

with open("requirements.txt", encoding="utf-8") as fp:
    requirements = fp.read().splitlines()
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()
version = {}
with open("longrefiner/version.py", encoding="utf8") as fp:
    exec(fp.read(), version)

setup(
    name="longrefiner",
    version=version["__version__"],
    packages=find_packages(),
    license="MIT License",
    long_description=long_description,
    long_description_content_type="text/markdown",
    package_data={"": ["*.yaml"]},
    include_package_data=True,
    install_requires=requirements,
    python_requires=">=3.10",
)
