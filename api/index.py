"""
api/index.py
============
Vercel Serverless Function entrypoint for SentryICU FastAPI backend.
Directs all incoming /api/* requests to the production FastAPI instance.
"""

import os
import sys

# Append project root to sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.api import app

# Vercel ASGI handler
app = app
