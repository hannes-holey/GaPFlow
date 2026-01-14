# Contributing guidelines

Thank you for your interest in contributing! We welcome contributions in the form
of bug reports, feature and pull requests. Please follow the instructions below.

### Reporting Issues
- Use the [issue tracker](https://github.com/hannes-holey/GaPFlow/issues) to report
  bugs, request features, or suggest improvements.
- Clearly describe the problem and include steps to reproduce when applicable.

### Submitting Pull Requests
- Fork the repository and create a feature branch.
- Ensure your code is well-documented and follows the existing style.
- Include tests for new functionality when possible.
- Update documentation as needed.
- Submit a pull request with a clear description of your changes.

### Code style
We follow [PEP-8](https://peps.python.org/pep-0008/) with a few exceptions.
Docstrings should follow the [numpydoc](https://numpydoc.readthedocs.io/en/latest/
format.html) style. Before committing, please run a linter such as `flake8` to
ensure your changes meet the project's style standards.

### Pre-commit hooks
We use [pre-commit](https://pre-commit.com) to ensure code and notebooks stay
clean. To set up locally:

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

### Third-party code and licensing
This repository includes third-party components. The following notes summarize
the most important license information for contributors and redistributors.

#### LAMMPS
- Location in this repository: `lammps/` and the vendored Python bindings under
  `GaPFlow/_vendor/lammps/` (the vendored bindings include local modifications
  to the Python interface).
- License: GNU General Public License version 2 (GPLv2). See `lammps/LICENSE`
  and the repository `COPYING` file for the full text.
- Implication: If you distribute a packaged artifact (sdist or wheel) that
  includes the LAMMPS sources or vendored LAMMPS Python bindings, the combined
  distribution is subject to the terms of the GPLv2. Recipients must be
  granted the rights required by the GPLv2, and the full license text
  (`COPYING`) must be included in distributed artifacts.

#### Upstream LAMMPS sources and bundled components
The LAMMPS source tree contains various bundled components. To avoid
inadvertently redistributing bundled third-party source code, the project's
source-distribution settings are configured so that the published source
distribution does not include third-party library source folders by default.
If you need the full upstream LAMMPS sources including bundled third-party
libraries and their license texts, consult the official LAMMPS distribution
at https://lammps.org or inspect the `lammps/` directory in this repository.

##### Contributor responsibilities
- When adding, upgrading, or removing bundled third-party code, update this
  `Third-party code and licensing` section in `CONTRIBUTING.md` to describe the
  component, its location, and its license.
- If you modify or update vendored third-party code (including the files under
  `GaPFlow/_vendor/lammps/`), preserve all original copyright and license
  headers.
- When adding new third-party code, include the upstream license text in the
  new files or a dedicated license file in the component directory.
- Be aware that distributing packaged artifacts (sdist/wheel) that include
  GPLv2-covered files (for example the vendored LAMMPS bindings) imposes GPLv2
  obligations on the combined distribution. Consult the maintainers if you are
  unsure whether a change will affect the project's distribution license.

##### Preservation of notices
All original copyright and license notices in third-party files must be
retained. When redistributing, do not remove or alter these notices.

### Code of Conduct

We aim to maintain a welcoming, respectful, and inclusive community.  
Please be courteous, constructive, and considerate in all interactions.  
Harassment or disrespectful behavior is not tolerated.

By contributing, you agree to follow this Code of Conduct.