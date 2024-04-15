from localizations.abstract_localizer import AbstractLocalizer
from masks import AbstractMask, CausalGraphMask
from tasks.task import Task

from cb_utils.mask_utils import get_masks_from_eap_exp
from localizations.eap.eap_wrapper import EAP

class EAPLocalizer(AbstractLocalizer):

    def __init__(self, model, task):
        self.model = model
        self.task = task

    def get_mask(self, batch=20, threshold=0.0005, filter_neox=True) -> AbstractMask:
        
        if type(self.task).__name__ == "InductionTask":
            clean_dataset = self.task.clean_data
            corr_dataset = self.task.corr_data
            eap_metric = self.task.get_acdcpp_metric()

            graph = EAP(
                self.model,
                clean_dataset,
                corr_dataset,
                eap_metric,
                upstream_nodes=["mlp", "head"],
                downstream_nodes=["mlp", "head"],
                batch_size=batch,
                clean_answers=None,
            )
        elif type(self.task).__name__ == "IOITask":
            clean_dataset = self.task.clean_data
            corr_dataset = self.task.corr_data
            eap_metric = self.task.get_acdcpp_metric(self.model)

            graph = EAP(
                self.model,
                clean_dataset.toks,
                corr_dataset.toks,
                eap_metric,
                upstream_nodes=["mlp", "head"],
                downstream_nodes=["mlp", "head"],
                batch_size=batch,
                clean_answers=None,
            )

        (
            acdcpp_nodes,
            acdcpp_edges,
            acdcpp_mask_dict,
            acdcpp_weight_mask_attn_dict,
            acdcpp_weight_mask_mlp_dict,
        ) = get_masks_from_eap_exp(
            graph, threshold=0.0005, num_layers=self.model.cfg.n_layers, num_heads=self.model.cfg.n_heads, filter_neox=filter_neox
        )

        return CausalGraphMask(
            nodes_set=acdcpp_nodes,
            edges_set=acdcpp_edges,
            ct_mask_dict=acdcpp_mask_dict,
            ct_weight_mask_attn_dict=acdcpp_weight_mask_attn_dict,
            ct_weight_mask_mlp_dict=acdcpp_weight_mask_mlp_dict,
        )
