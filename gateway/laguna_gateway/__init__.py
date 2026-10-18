"""DGX-side OpenAI-compatible gateway for Botio and Laguna S 2.1."""

from .app import create_app
from .config import Settings

__all__ = ["Settings", "create_app"]
