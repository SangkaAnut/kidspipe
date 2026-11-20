"""KiDS-Legacy (DR5) COSEBIs validation pipeline (Wright et al. 2025, A&A 703, A158).

Self-contained alternative to the KV450 model: 6 tomographic bins, CAMB HMcode-2020 (+baryon feedback log T_AGN),
mass-dependent NLA intrinsic alignments, correlated photo-z shifts, 2-300 arcmin COSEBIs modes 1-6.
See docs/interface_notes/legacy.md.
"""
from .data import LegacyData, LegacyCosebisLikelihood, load_chain, chain_samples
from .ia import ia_amplitudes
from .model import LegacyCosebisModel, LegacyOptions

__all__ = ["LegacyData", "LegacyCosebisLikelihood", "load_chain", "chain_samples", "ia_amplitudes",
           "LegacyCosebisModel", "LegacyOptions"]
