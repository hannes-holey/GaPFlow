# Contributing guidelines

Thank you for your interest in contributing! We welcome contributions in the form of bug reports, feature and pull requests. Please follow the instructions below.

### Reporting Issues
- Use the [issue tracker](https://github.com/hannes-holey/GaPFlow/issues) to report bugs, request features, or suggest improvements.
- Clearly describe the problem and include steps to reproduce when applicable.

### Submitting Pull Requests
- Fork the repository and create a feature branch.
- Ensure your code is well-documented and follows the existing style.
- Include tests for new functionality when possible.
- Update documentation as needed.
- Submit a pull request with a clear description of your changes.

### Code style
We follow [PEP-8](peps.python.org/pep-0008/) with a few exceptions.
Docstrings should follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html) style.
Before committing, please run a linter such as `flake8` to ensure your changes meet the project's style standards.

### Pre-commit hooks
We use [pre-commit](https://pre-commit.com) to ensure code and notebooks stay clean.
To set up locally:
```bash
pip install pre-commit nb-clean
pre-commit install
```

### Building the documentation
A Sphinx-generated documentation can be built locally with
```bash
cd doc
sphinx-apidoc -o . ../GaPFlow
make html
```

### Code of Conduct

We aim to maintain a welcoming, respectful, and inclusive community.  
Please be courteous, constructive, and considerate in all interactions.  
Harassment or disrespectful behavior is not tolerated.

By contributing, you agree to follow this Code of Conduct.