# How to contribute to PyRASA

We always welcome contributions to PyRASA.

## General rules

## Setting up the development environment
We have decided to use the following tools:

1. [Pixi](https://pixi.sh) for dependency management.
2. [Ruff](https://docs.astral.sh/ruff/) for linting and formatting.
3. [Hatch](https://hatch.pypa.io/) for packaging and distribution.

### Install the software
The first thing to do is to install [Pixi](https://pixi.sh) which manages all the rest for us.

Check the [installation guide on their website](https://pixi.sh/latest/#installation)

Afterwards, just run `pixi install` in the root folder of the repository and
it's going to take care of the rest.

This is going to create a so-called "environment", i.e. a folder in which all
the dependencies are installed. To use the environment, you need to run `pixi shell`.

### Optional: Install the precommit hook
We have a precommit hook that runs the linter and the formatter before every commit.
To install it, run `pre-commit install`.

## How to do common tasks
### Add dependencies for the project

There are two kinds of dependencies: runtime and development dependencies. Both
are managed in the `pyproject.toml` file.

* Runtime dependencies are added to the `[project]` -> `dependencies` section.
* Development dependencies can be found in the `[tool.pixi.dependencies]` section.
  * You can use the `pixi add` command to add a new development dependency.

After you have edited the `pyproject.toml` file, run `pixi install` to install the new dependencies.

### Linting and formatting
Linting and formatting are done with [Ruff](https://docs.astral.sh/ruff/).
These rules are going to be enforced by CI in the future.

If you want to check your code manually, just run `pixi run lint`.

### Building and distributing the package
We use [Hatch](https://hatch.pypa.io/) to build and distribute the package. It
is also used to manage the versioning of the package.

Eventually, building and distributing the package will be done automatically by CI.
But if you want to test it, you can just use `hatch build` and it is going to
create a wheel file in the `dist` folder.

### Managing version
We use [Hatch](https://hatch.pypa.io/) to manage the version of the package.

The single source of truth for the version is found in the `pyrasa/__version__.py` file.

You can use hatch to view the current version with `hatch version`. If you want
to create a new version, hatch increments the version number for you and also creates
a tag for you.

* `hatch version dev` creates a new development version.
* `hatch version fix` creates a new patch version.
* `hatch version minor` creates a new minor version.

Just make sure to also push the tag to the repository with `git push --tags`.

