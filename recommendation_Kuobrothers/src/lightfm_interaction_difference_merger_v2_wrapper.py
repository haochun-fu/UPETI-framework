# -*- coding: utf-8 -*-
"""LightFMInteractionDifferenceMergerV2Wrapper class."""
from dataloader_lightfm_interaction_difference_merger_v2 import \
    LightFMInteractionDifferenceMergerV2DataLoader
from lightfm_wrapper import LightFMWrapper


__author__ = 'haochun.fu'
__date__ = '2020-06-27'


class LightFMInteractionDifferenceMergerV2Wrapper(LightFMWrapper):
    """Wrapper of LightFM interaction difference merger version 2."""


    def get_dataloader_class(
        self
    ) -> LightFMInteractionDifferenceMergerV2DataLoader:
        """Get dataloader class.

        Returns:
            LightFMInteractionDifferenceMergerV2DataLoader: 
                LightFMInteractionDifferenceMergerV2DataLoader class.
        """
        return LightFMInteractionDifferenceMergerV2DataLoader
