# -*- coding: utf-8 -*-
"""LightFMInteractionDifferenceEnsemblerWrapper class."""
from dataloader_lightfm_interaction_difference import \
    LightFMInteractionDifferenceDataLoader
from interaction_difference_wrapper import \
    MatrixEnsemblerInteractionDifferenceWrapper
from lightfm_wrapper import LightFMWrapper


__author__ = 'haochun.fu'
__date__ = '2020-02-22'


class LightFMInteractionDifferenceEnsemblerWrapper(
    MatrixEnsemblerInteractionDifferenceWrapper):
    """Wrapper of LightFM interaction difference ensembler."""


    def get_dataloader_class(self) -> LightFMInteractionDifferenceDataLoader:
        """Get dataloader class.

        Returns:
            LightFMInteractionDifferenceDataLoader: 
                LightFMInteractionDifferenceDataLoader class.
        """
        return LightFMInteractionDifferenceDataLoader

    def get_model_wrapper_class(self) -> LightFMWrapper:
        """Get model wrapper class.

        Return:
            LightFMWrapper: LightFM wrapper class.
        """
        return LightFMWrapper 
