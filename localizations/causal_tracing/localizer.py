from platform import node
from localizations.abstract_localizer import AbstractLocalizer
from localizations.causal_tracing.causal_tracing import (
    causal_tracing_induction,
    causal_tracing_ioi,
)
from masks.masks import CausalGraphMask, MaskType

from cb_utils.mask_utils import get_masks_from_ct_nodes


# TODO: Move functions from causal_tracing.py to this class
class CausalTracingLocalizer(AbstractLocalizer):
    def __init__(self, model, task):
        self.model = model
        self.task = task

    def get_mask(self, model, task, threshold=0.0005):
        """
        Get the mask (of nodes) for the model and task.

        Supported Tasks:
            - InductionTask
            - IOITask
        """

        if type(task).__name__ == "InductionTask":
            mask = causal_tracing_induction(model, task)

        elif type(task).__name__ == "IOITask":
            mask = causal_tracing_ioi(model, task)

        ct_keys = list(mask.keys())
        ct_keys_above_threshold = [k for k in ct_keys if mask[k] > threshold]
        ct_keys_above_threshold
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
