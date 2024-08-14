from abc import ABC, abstractmethod
from enum import Enum
from attr import dataclass

import pickle

class MaskType(Enum):
    CAUSAL_GRAPH_MASK = 1

class AbstractMask(ABC):

    @abstractmethod
    def save(self, path):
        path

@dataclass
class CausalGraphMask(AbstractMask):
    nodes_set: set
    edges_set: set
    ct_mask_dict: dict
    ct_weight_mask_attn_dict: dict
    ct_weight_mask_mlp_dict: dict
    mask_type: MaskType = MaskType.CAUSAL_GRAPH_MASK

    # Save to pkl 
    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(
                (
                    self.nodes_set, 
                    self.edges_set, 
                    self.ct_mask_dict, 
                    self.ct_weight_mask_attn_dict, 
                    self.ct_weight_mask_mlp_dict 
                ), 
                f
            )

