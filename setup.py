from setuptools import setup, find_packages

# Function to read requirements from requirements.txt
def parse_requirements(filename):
    lines = (line.strip() for line in open(filename))
    # Filter out comments and empty lines
    requirements = [line for line in lines if line and not line.startswith("#")]
    return requirements

setup(
    name='obsidian-ai-search',
    version='0.1.0', # Initial version
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'click>=8.0',
        'numpy>=1.21',
        'sentence-transformers>=2.2.0', # Check latest compatible version
        'faiss-cpu>=1.7.0', # Check latest compatible version
        'watchdog>=2.1.0',
        'rich>=10.0',
        'tomli>=1.1.0; python_version < "3.11"', # For older Python TOML parsing
        'python-dotenv>=0.19.0', # Added for .env support
        'google-generativeai>=0.3.0', # For Gemini RAG
        'prompt_toolkit>=3.0', # For REPL history/suggestions
        'pyyaml>=5.4', # If using YAML for config or other data
        'psutil>=5.8.0', # For system monitoring (optional, e.g., memory usage)
        # Add FastAPI and uvicorn if the API server is core
        # 'fastapi>=0.70.0',
        # 'uvicorn[standard]>=0.15.0',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'pytest-cov>=4.0.0',
            'black>=23.0.0',
            'isort>=5.10.0',
            'mypy>=1.0.0',
            'flake8>=5.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'obsidian-search = obsidian_ai_search.cli:cli', # Command name = module:function
        ],
    },
    # Metadata
    author='Obsidian AI Search Contributors',
    author_email='example@example.com',
    description='Locally index and search Obsidian notes using embeddings.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/obsidian-ai-search', # Replace with your repo URL if applicable
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License', # Choose appropriate license
        'Operating System :: OS Independent',
        'Development Status :: 3 - Alpha',
        'Topic :: Utilities',
        'Environment :: Console',
    ],
    python_requires='>=3.8', # Specify minimum Python version
) 