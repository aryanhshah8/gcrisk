from setuptools import setup, find_packages

setup(
    name='gcrisk',
    version='1.0.0',
    author='Aryan Hemendra Shah',
    author_email='ahs222@miami.edu',
    url='https://github.com/aryanhshah8/gcrisk',
    description='Open-source GCR mission dosimetry and organ-risk pipeline',
    license='MIT',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=[line.strip() for line in open('requirements.txt')],
    entry_points={
        'console_scripts': [
            'gcrisk-dose=gcrisk.cli:main',
        ],
    },
    extras_require={
        'dev': [
            'pytest>=7.4',
            'hypothesis>=6.0',
            'sphinx>=7.0',
            'sphinx-rtd-theme>=1.3',
        ],
        'notebooks': [
            'jupyter>=1.0.0',
        ],
    },
)
