from setuptools import setup, find_packages

setup(
    name="cancer-research-assistant",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    author="DHRUMIL PATEL, YSHUSSAIN, Sahil Kasliwal ",
    author_email="dhrumil2510@gmail.com",
    description="A comprehensive cancer research analysis tool",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/DAMG7250-Team1/Hackathon",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
) 