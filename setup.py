from setuptools import find_packages, setup

# load readme
with open("README.md", 'r') as f:
    long_description = f.read()

setup(name="torch-liter",
      version="0.0.0",
      author="Chenchao Zhao",
      author_email="chenchao.zhao@gmail.com",
      description="A light weight training tool for pytorch projects.",
      py_modules=["liter"],
      long_description=long_description,
      long_description_content_type="text/markdown",
      install_requires=["torch", "numpy"],
      license="MIT")