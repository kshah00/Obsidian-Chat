# Contributing to Obsidian AI Search

Thank you for considering contributing to Obsidian AI Search! This document provides guidelines and instructions for contributing.

## Development Setup

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/obsidian-ai-search.git
   cd obsidian-ai-search
   ```
3. Set up the development environment:
   ```bash
   conda env create -f environment.yml
   conda activate obsidian-search
   pip install -e ".[dev]"  # Install development dependencies
   ```

## Code Style

We follow PEP 8 style guidelines for Python code. Please ensure your code is formatted accordingly.

## Pull Request Process

1. Create a new branch for your feature or bugfix
2. Make your changes and add tests if applicable
3. Ensure all tests pass:
   ```bash
   pytest
   ```
4. Update documentation if necessary
5. Submit a pull request with a clear description of the changes

## Commit Messages

Please use clear and meaningful commit messages. Follow the conventional commits format when possible:

- `feat`: A new feature
- `fix`: A bug fix
- `docs`: Documentation changes
- `style`: Code style changes (formatting, etc.)
- `refactor`: Code refactoring
- `test`: Adding or updating tests
- `chore`: Maintenance tasks

Example: `feat: add support for custom chunking strategies`

## Testing

When adding new features, please include appropriate tests. We aim for good test coverage to ensure reliability.

## License

By contributing, you agree that your contributions will be licensed under the project's MIT License. 