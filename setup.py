#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

requirements = [
    "ConfigParser==5.3.0",
    "openai==0.27.4",
    "dftotext==2.2.2",
    "tiktoken==0.3.3",
    "Flask==2.2.3",
    "Flask-Cors==3.0.10",
    "python-dotenv==1.0.0",
]

test_requirements = [
    "pytest",
    "pytest-timeout",
]

setup(
    name='document_summarizer',
    version='0.0.0',
    description="Using GPT4 to create a document summarizer",
    long_description=readme,
    author="Jack King",
    author_email='jackking@mit.edu',
    url='https://github.com/JGKing88/document_summarizer',
    packages=find_packages(exclude=['tests']),
    install_requires=requirements,
    license="MIT license",
    zip_safe=False,
    test_suite='tests',
    tests_require=test_requirements
)
