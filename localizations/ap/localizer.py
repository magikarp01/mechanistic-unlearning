from localizations.abstract_localizer import AbstractLocalizer
from masks import AbstractMask, CausalGraphMask
from tasks.task import Task

from cb_utils.mask_utils import get_masks_from_ct_nodes
from localizations.ap.ap_wrapper import AP

class APLocalizer(AbstractLocalizer):

    def __init__(self, model, task):
        self.model = model
        self.task = task

    def get_ap_graph(self, batch=20):
        '''

            Similar to get_mask, but returns the AP Nodes instead

            AP nodes can be converted to CausalGraphMask using get_masks_from_ct_nodes
            Since AP and CT follow the same node based localization pattern

            someone pls pls change the name of the function from get_masks_from_ct_nodes to get_masks_from_nodes
        '''

        if type(self.task).__name__ == "InductionTask":
            clean_dataset = self.task.clean_data
            corr_dataset = self.task.corr_data
            eap_metric = self.task.get_acdcpp_metric()

            nodes = AP(
                self.model,
                clean_dataset,
                corr_dataset,
                eap_metric,
                batch_size=batch,
                clean_answers=None,
            )
        elif type(self.task).__name__ == "IOITask":
            clean_dataset = self.task.clean_data
            corr_dataset = self.task.corr_data
            eap_metric = self.task.get_acdcpp_metric(self.model)

            nodes = AP(
                self.model,
                clean_dataset.toks,
                corr_dataset.toks,
                eap_metric,
                batch_size=batch,
                clean_answers=None,
            )
        elif type(self.task).__name__ == "SportsFactsTask":
            clean_dataset = self.task.clean_data
            corr_dataset = self.task.corr_data
            eap_metric = self.task.get_acdcpp_metric()

            nodes = AP(
                self.model,
                clean_dataset.toks,
                corr_dataset.toks,
                eap_metric,
                batch_size=batch,
                clean_answers=self.task.clean_answer_toks,
                wrong_answers=self.task.clean_wrong_toks,
            )
        return nodes 

    def get_mask(self, batch=20, threshold=0.0005) -> AbstractMask:
        nodes = self.get_exp_graph(batch=batch)        

        ap_keys = list(nodes.keys())
        ap_keys_above_threshold = [k for k in ap_keys if nodes[k] > threshold]
        (
            nodes_set,
            edges_set,
            ct_mask_dict,
            ct_weight_mask_attn_dict,
            ct_weight_mask_mlp_dict,
        ) = get_masks_from_ct_nodes(ap_keys_above_threshold)
        return CausalGraphMask(
            nodes_set=nodes_set,
            edges_set=edges_set,
            ct_mask_dict=ct_mask_dict,
            ct_weight_mask_attn_dict=ct_weight_mask_attn_dict,
            ct_weight_mask_mlp_dict=ct_weight_mask_mlp_dict,
        )
