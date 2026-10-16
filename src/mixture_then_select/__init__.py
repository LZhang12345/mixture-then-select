"""Mixture-then-Select allocation and subset selection."""

from .core import Selection, allocate, choose_mixture, select, selection_hash

__all__ = ["Selection", "allocate", "choose_mixture", "select", "selection_hash"]
__version__ = "0.1.0"
