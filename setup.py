from setuptools import find_packages, setup


setup(
        name='embedding_studio',
        version='1.0.0',
        packages=find_packages(),
        package_data={'': ['*.yaml', '*.txt']},
        include_package_data=True
)