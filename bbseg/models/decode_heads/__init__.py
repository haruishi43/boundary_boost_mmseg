#!/usr/bin/env python3

# SBCB auxiliaries
from .casenet_head import GeneralizedCASENetHead, AuxCASENetHead
from .dff_head import GeneralizedDFFHead, AuxDFFHead
from .dds_head import GeneralizedDDSHead, AuxDDSHead

# Deocde heads
from .mod_sep_aspp_head import ModDepthwiseSeparableASPPHead


__all__ = [
    "GeneralizedCASENetHead",
    "AuxCASENetHead",
    "GeneralizedDFFHead",
    "AuxDFFHead",
    "GeneralizedDDSHead",
    "AuxDDSHead",
    "ModDepthwiseSeparableASPPHead",
]
