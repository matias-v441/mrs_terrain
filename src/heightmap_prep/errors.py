"""Error hierarchy for the heightmap preparation library.

Every failure mode listed in the specification maps onto one of these types so
callers can react to categories of problems rather than parsing messages.
"""

from __future__ import annotations


class HeightmapPrepError(Exception):
    """Base class for every error raised by this library."""


# --- configuration -------------------------------------------------------


class ConfigError(HeightmapPrepError):
    """A world configuration file is missing, malformed or self-inconsistent."""


# --- coordinate reference systems / PROJ ---------------------------------


class CrsError(HeightmapPrepError):
    """A CRS could not be built, matched or used as required."""


class UnsupportedSourceCrsError(CrsError):
    """A source raster is not in the horizontal CRS the pipeline expects."""


class UnexpectedVerticalDatumError(CrsError):
    """A source raster does not carry the vertical datum the pipeline expects."""


class TransformationUnavailableError(CrsError):
    """PROJ cannot provide a non-ballpark operation for the requested datums."""


class MissingGridError(TransformationUnavailableError):
    """The operation PROJ selected needs grid files that are not installed."""

    def __init__(self, message: str, missing_grids: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.missing_grids = tuple(missing_grids)


class NonFiniteTransformError(CrsError):
    """A vertical transformation produced ``inf``/``nan`` for valid input."""


# --- acquisition ---------------------------------------------------------


class SourceError(HeightmapPrepError):
    """A height source could not deliver the requested raster."""


class SourceHttpError(SourceError):
    """The remote service failed after all configured retries."""


class OutOfCoverageError(SourceError):
    """The requested area lies outside the source product's coverage."""


class MalformedRasterError(SourceError):
    """A raster could not be opened or is structurally unusable."""


class ResolutionMismatchError(SourceError):
    """A returned raster does not have the requested pixel size."""


# --- output --------------------------------------------------------------


class OutputConflictError(HeightmapPrepError):
    """An output already exists and overwriting was not permitted."""


class ValidationError(HeightmapPrepError):
    """A generated dataset failed validation."""
