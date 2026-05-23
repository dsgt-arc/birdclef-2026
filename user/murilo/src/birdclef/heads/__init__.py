"""Classification heads for audio models."""

from .sed_head import AttHead, AttHeadLogitSpace, GeM, GeMFreqPool

__all__ = ["AttHead", "AttHeadLogitSpace", "GeM", "GeMFreqPool"]
