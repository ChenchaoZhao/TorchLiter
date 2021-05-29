from setuptools import find_packages, setup

# load readme
with open("README.md", "r") as f:
    long_description = f.read()


def version():
    import liter

    return liter.__version__


setup(
    name="torchliter",
    version=version(),
    author="Chenchao Zhao",
    author_email="chenchao.zhao@gmail.com",
    description="A lightweight training tool for pytorch projects.",
    packages=find_packages(exclude=["tests"]),
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=["torch", "numpy"],
    license="MIT",
)
