"""
Server app entry point for OpenEnv deployment.
Imports and exposes the FastAPI app from the main server module.
"""

# Import the app from the root server module
# This wrapper exists to support standard OpenEnv deployment patterns
import sys
from pathlib import Path

# Add parent directory to path so we can import from root server.py
sys.path.insert(0, str(Path(__file__).parent.parent))

from server import app

__all__ = ["app"]
