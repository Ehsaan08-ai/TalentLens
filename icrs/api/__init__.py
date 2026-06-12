"""ICRS HTTP API package (Task 17.1).

Exposes :func:`create_app`, the FastAPI app factory serving the asynchronous
ranking pipeline. See :mod:`icrs.api.app` for the endpoint contract and the
PoC-only / unauthenticated security note.
"""

from icrs.api.app import create_app

__all__ = ["create_app"]
