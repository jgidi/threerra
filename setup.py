import setuptools

with open("README.md", "r", encoding="utf-8") as readme:
    description = readme.read()

setuptools.setup(
    name='threerra',
    version='0.0.1',
    author='Jorge Gidi',
    author_email='jorgegidi@gmail.com',
    description='A Qiskit module for three-level systems',
    long_description=description,
    long_description_content_type="text/markdown",
    url='https://github.com/jgidi/threerra',
    license='Apache Licence 2.0',
    packages=['threerra'],
    install_requires=['qiskit', 'numpy', 'scipy', 'matplotlib', 'sklearn'],
)
