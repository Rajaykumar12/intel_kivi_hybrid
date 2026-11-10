"""Offline export of a model's decode step for the Python-free runtime
(Phase 5, Route A). Python is needed only here, once per model; the exported
programs run with the C++ engine registered as id 0."""

from .decoder import RUNTIME_ENGINE_ID, KiviExportDecoder, export_programs

__all__ = ["RUNTIME_ENGINE_ID", "KiviExportDecoder", "export_programs"]
