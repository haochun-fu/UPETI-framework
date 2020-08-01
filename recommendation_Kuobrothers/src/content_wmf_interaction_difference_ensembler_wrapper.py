# -*- coding: utf-8 -*-
"""ContentWMFInteractionDifferenceEnsemblerWrapper class."""
from dataloader_content_wmf_interaction_difference import \
    ContentWMFInteractionDifferenceDataLoader
from interaction_difference_wrapper import \
    MatrixEnsemblerInteractionDifferenceWrapper
from content_wmf_wrapper import ContentWMFWrapper


__author__ = 'haochun.fu'
__date__ = '2020-02-25'


class ContentWMFInteractionDifferenceEnsemblerWrapper(
    MatrixEnsemblerInteractionDifferenceWrapper):
    """Wrapper of ContentWMF interaction difference ensembler."""


    def get_dataloader_class(self) -> ContentWMFInteractionDifferenceDataLoader:
        """Get dataloader class.

        Returns:
            ContentWMFInteractionDifferenceDataLoader: 
                ContentWMFInteractionDifferenceDataLoader class.
        """
        return ContentWMFInteractionDifferenceDataLoader

    def get_model_wrapper_class(self) -> ContentWMFWrapper:
        """Get model wrapper class.

        Return:
            ContentWMFWrapper: ContentWMF wrapper class.
        """
        return ContentWMFWrapper 
