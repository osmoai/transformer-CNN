from setuptools import setup, find_packages

setup(
    name='transformer-cnn',
    version='0.6.0',
    author='guillaume godin',
    author_email='guillaume@osmo.ai',
    description='Transformer-CNN',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/osmoai/transformer-CNN',
    packages=find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.11',
    install_requires=[
        'tensorflow==2.15.1','rdkit==2023.09.3','numpy==1.26.4','scikit-learn==1.3.2','pandas'
    ],
    entry_points={
        'console_scripts': [
            'transformer-cnn=transformer_cnn.run:main',
        ],
    },
)
