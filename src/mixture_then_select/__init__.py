"""Mixture-then-Select with the fixed lambda-0.50 allocation rule."""

from .core import Selection, allocate, choose_mixture, select, selection_hash

__all__ = ["Selection", "allocate", "choose_mixture", "select", "selection_hash"]
__version__ = "0.1.0"
