"""Classes implementing generalized linear regression estimators."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("sparse-lm")
except PackageNotFoundError:
    # package is not installed
    __version__ = ""
