"""
Taxonomy and tagging subpackage for longhorizon_guard.
"""

from .categories import DEFAULT_TAGS
from .tagger import tag_failures

__all__ = ["DEFAULT_TAGS", "tag_failures"]
