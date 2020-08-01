# -*- coding: utf-8 -*-
"""LightFMInteractionDifferenceEnsemblerWrapper class."""
from lightfm_interaction_difference_ensembler_dataloader import \
    LightFMInteractionDifferenceEnsemblerDataLoader
from interaction_difference_wrapper import \
    MatrixEnsemblerInteractionDifferenceWrapper
from lightfm_wrapper import LightFMWrapper


__author__ = 'haochun.fu'
__date__ = '2020-07-06'


class LightFMInteractionDifferenceEnsemblerWrapper(
    MatrixEnsemblerInteractionDifferenceWrapper):
    """Wrapper of LightFM interaction difference ensembler."""


    def get_dataloader_class(
        self
    ) -> LightFMInteractionDifferenceEnsemblerDataLoader:
        """Get dataloader class.

        Returns:
            LightFMInteractionDifferenceEnsemblerDataLoader: 
                LightFMInteractionDifferenceEnsemblerDataLoader class.
        """
        return LightFMInteractionDifferenceEnsemblerDataLoader

    def get_model_wrapper_class(self) -> LightFMWrapper:
        """Get model wrapper class.

        Return:
            LightFMWrapper: LightFM wrapper class.
        """
        return LightFMWrapper 
