from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version('finn')
except PackageNotFoundError:
    __version__ = 'unknown'
