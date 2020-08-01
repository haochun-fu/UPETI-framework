# -*- coding: utf-8 -*-
"""ContentWMFInteractionDifferenceMergerV2Wrapper class."""
from dataloader_content_wmf_interaction_difference_merger_v2 import \
    ContentWMFInteractionDifferenceMergerV2DataLoader
from content_wmf_wrapper import ContentWMFWrapper


__author__ = 'haochun.fu'
__date__ = '2020-06-27'


class ContentWMFInteractionDifferenceMergerV2Wrapper(ContentWMFWrapper):
    """Wrapper of ContentWMF interaction difference merger version 2."""


    def get_dataloader_class(
        self
    ) -> ContentWMFInteractionDifferenceMergerV2DataLoader:
        """Get dataloader class.

        Returns:
            ContentWMFInteractionDifferenceMergerV2DataLoader: 
                ContentWMFInteractionDifferenceMergerV2DataLoader class.
        """
        return ContentWMFInteractionDifferenceMergerV2DataLoader
