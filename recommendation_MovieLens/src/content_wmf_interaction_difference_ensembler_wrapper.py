# -*- coding: utf-8 -*-
"""ContentWMFInteractionDifferenceEnsemblerWrapper class."""
from content_wmf_interaction_difference_ensembler_dataloader import \
    ContentWMFInteractionDifferenceEnsemblerDataLoader
from interaction_difference_wrapper import \
    MatrixEnsemblerInteractionDifferenceWrapper
from content_wmf_wrapper import ContentWMFWrapper


__author__ = 'haochun.fu'
__date__ = '2020-07-06'


class ContentWMFInteractionDifferenceEnsemblerWrapper(
    MatrixEnsemblerInteractionDifferenceWrapper):
    """Wrapper of ContentWMF interaction difference ensembler."""


    def get_dataloader_class(
        self
    ) -> ContentWMFInteractionDifferenceEnsemblerDataLoader:
        """Get dataloader class.

        Returns:
            ContentWMFInteractionDifferenceEnsemblerDataLoader: 
                ContentWMFInteractionDifferenceEnsemblerDataLoader class.
        """
        return ContentWMFInteractionDifferenceEnsemblerDataLoader

    def get_model_wrapper_class(self) -> ContentWMFWrapper:
        """Get model wrapper class.

        Return:
            ContentWMFWrapper: ContentWMF wrapper class.
        """
        return ContentWMFWrapper 
