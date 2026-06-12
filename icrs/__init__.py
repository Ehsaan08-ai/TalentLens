"""Intelligent Candidate Ranking System (ICRS).

A five-layer pipeline that ranks job candidates through semantic understanding,
behavioral inference, and multi-signal scoring rather than keyword matching.

This package is organized into:
    icrs.models       - typed data models (RequirementVector, profiles, results)
    icrs.providers    - abstract provider interfaces + concrete swappable backends
    icrs.pipeline     - JD decomposition, enrichment, embedding, orchestration
    icrs.scoring      - hybrid scoring engine (hard filter, dense/sparse, fusion)
    icrs.persistence  - PostgreSQL + pgvector / Qdrant repositories
    icrs.ui           - Streamlit PoC dashboard
"""

__version__ = "0.1.0"
