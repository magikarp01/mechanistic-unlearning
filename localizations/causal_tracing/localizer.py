from platform import node
from localizations.abstract_localizer import AbstractLocalizer
from localizations.causal_tracing.causal_tracing import (
    causal_tracing_induction,
    causal_tracing_ioi,
    causal_tracing_sports
)
from masks import CausalGraphMask, MaskType

from cb_utils.mask_utils import get_masks_from_ct_nodes


# TODO: Move functions from causal_tracing.py to this class
class CausalTracingLocalizer(AbstractLocalizer):
    def __init__(self, model, task):
        self.model = model
        self.task = task

    def get_ct_mask(self, batch_size=5, gemma2=False):
        '''
            Almost like get_mask, but returns the keys of the causal tracing dictionary
            You can get the mask by calling get_masks_from_ct_nodes(keys)

            Useful if you want to test different thresholds since you only have to run this once
        '''
        if type(self.task).__name__ == "InductionTask":
            mask = causal_tracing_induction(self.model, self.task)

        elif type(self.task).__name__ == "IOITask":
            mask = causal_tracing_ioi(self.model, self.task)
        elif type(self.task).__name__ == "SportsFactsTask":
            mask = causal_tracing_sports(self.model, self.task, batch_size, gemma2=gemma2)

        return mask

    def get_mask(self, batch_size=5, threshold=0.0005):
        """
        Get the mask (of nodes) for the model and task.

        Supported Tasks:
            - InductionTask
            - IOITask
        """

        if type(self.task).__name__ == "InductionTask":
            mask = causal_tracing_induction(self.model, self.task)

        elif type(self.task).__name__ == "IOITask":
            mask = causal_tracing_ioi(self.model, self.task)
        elif type(self.task).__name__ == "SportsFactsTask":
            mask = causal_tracing_sports(self.model, self.task, batch_size)

        ct_keys = list(mask.keys())
        ct_keys_above_threshold = [k for k in ct_keys if mask[k] > threshold]
        (
            nodes_set,
            edges_set,
            ct_mask_dict,
            ct_weight_mask_attn_dict,
            ct_weight_mask_mlp_dict,
        ) = get_masks_from_ct_nodes(ct_keys_above_threshold)
        return CausalGraphMask(
            nodes_set=nodes_set,
            edges_set=edges_set,
            ct_mask_dict=ct_mask_dict,
            ct_weight_mask_attn_dict=ct_weight_mask_attn_dict,
            ct_weight_mask_mlp_dict=ct_weight_mask_mlp_dict,
        )
