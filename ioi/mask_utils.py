from models import load_gpt2_weights, load_demo_gpt2, tokenizer
from data import retrieve_toxic_data, retrieve_owt_data, retrieve_toxic_data_low_loss, retrieve_toxic_filtered_data, FILTER_DEMO_LEN, CONTEXT_LENGTH
from inference import infer_batch_with_owt, infer_batch, prepare_fixed_demo, criterion
from torch.optim import AdamW
import torch
import pickle
import datasets
from tqdm import tqdm_notebook as tqdm
from itertools import cycle
# from eval import evaluate_model
from data import batch_text_to_tokens
import plotly.express as px

default_param_names = ['output_mask',
 'blocks.0.edge_mask_attentions',
 'blocks.0.edge_mask_mlp',
 'blocks.1.edge_mask_attentions',
 'blocks.1.edge_mask_mlp',
 'blocks.2.edge_mask_attentions',
 'blocks.2.edge_mask_mlp',
 'blocks.3.edge_mask_attentions',
 'blocks.3.edge_mask_mlp',
 'blocks.4.edge_mask_attentions',
 'blocks.4.edge_mask_mlp',
 'blocks.5.edge_mask_attentions',
 'blocks.5.edge_mask_mlp',
 'blocks.6.edge_mask_attentions',
 'blocks.6.edge_mask_mlp',
 'blocks.7.edge_mask_attentions',
 'blocks.7.edge_mask_mlp',
 'blocks.8.edge_mask_attentions',
 'blocks.8.edge_mask_mlp',
 'blocks.9.edge_mask_attentions',
 'blocks.9.edge_mask_mlp',
 'blocks.10.edge_mask_attentions',
 'blocks.10.edge_mask_mlp',
 'blocks.11.edge_mask_attentions',
 'blocks.11.edge_mask_mlp']

def load_mask_into_model(model, mask):
    # load in place
    mask_idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = mask[mask_idx].to(param.device)
            mask_idx += 1

def reset_mask(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = torch.ones_like(param.data).to(param.device)

def get_nodes_and_edges(mask_params, param_names=default_param_names, edge_0=True):
    """
    If edge_0 is True, then edges are between nodes with mask value 0. Else, edges are between nodes with mask value 1.
    Returns all_possible_nodes, nodes_with_edges, edges, mask_dict
    """
    # calculate which nodes will be in the graph
    connected_nodes = set()
    # add embed node at position
    # connected_nodes.add((-1, "embed"))
    n_heads = 12
    n_layers = 12

    # associate each node with a position
    all_possible_nodes = [(-1, "embed")]
    mask_dict = {}
    # empty tensor
    mask_dict["embed"] = torch.zeros(size=(0,))
    for idx in range(len(mask_params)):
        if "attention" in param_names[idx]:
            layer = int(param_names[idx].split(".")[1])
            for i in range(n_heads):
                all_possible_nodes.append((layer, f"a{layer}.{i}"))
                mask_dict[f"a{layer}.{i}"] = mask_params[idx][:,i].detach().cpu()
        elif "mlp" in param_names[idx]:
            layer = int(param_names[idx].split(".")[1])
            all_possible_nodes.append((layer, f"m{layer}"))
            mask_dict[f"m{layer}"] = mask_params[idx].detach().cpu()
    all_possible_nodes.append((n_heads, "output"))
    mask_dict["output"] = mask_params[0]

    # Calculate where edges are based on the mask
    # Edge between node i and node j if mask_dict[i][all_possible_nodes.index(j)] == 0
    edges = set()
    for i in range(len(all_possible_nodes)):
        for j in range(len(all_possible_nodes)):
            j_index = all_possible_nodes.index(all_possible_nodes[j])
            if j_index < len(mask_dict[all_possible_nodes[i][1]]) and mask_dict[all_possible_nodes[i][1]][all_possible_nodes.index(all_possible_nodes[j])] == (0 if edge_0 else 1):
                edges.add((all_possible_nodes[i], all_possible_nodes[j]))
    
    nodes_with_edges = set([node for edge in edges for node in edge])

    return all_possible_nodes, nodes_with_edges, edges, mask_dict


### Utils for ACDC to Mask
# acdcpp edges are in format 'blocks.1.attn.hook_result[:, :, 10]blocks.0.hook_mlp_in[:]', convert to format of ((1, 'a1.10'), (0, 'm0'))

def get_node_name(node_name, show_full_index=False):
    """Node name for use in pretty graphs"""

    def get_index(node_name_long):
        # Get the index by looking for number in brackets
        # e.g. blocks.1.attn.hook_result[:, :, 10] -> 10
        index = node_name_long.split("[")[-1].split("]")[0]
        index = index.split(", ")[-1]
        return int(index)

    if not show_full_index:
        name = ""
        qkv_substrings = [f"hook_{letter}" for letter in ["q", "k", "v"]]
        qkv_input_substrings = [f"hook_{letter}_input" for letter in ["q", "k", "v"]]

        # Handle embedz
        if "resid_pre" in node_name:
            assert "0" in node_name and not any([str(i) in node_name for i in range(1, 10)])
            name += "embed"
            layer = -1
            # if len(node.index.hashable_tuple) > 2:
            #     name += f"_[{node.index.hashable_tuple[2]}]"
            # return name

        elif "embed" in node_name:
            # name = "pos_embeds" if "pos" in node_name else "token_embeds"
            name = "embed"
            layer = -1

        # Handle q_input and hook_q etc
        elif any([node_name.endswith(qkv_input_substring) for qkv_input_substring in qkv_input_substrings]):
            relevant_letter = None
            for letter, qkv_substring in zip(["q", "k", "v"], qkv_substrings):
                if qkv_substring in node_name:
                    assert relevant_letter is None
                    relevant_letter = letter
            name += "a" + node_name.split(".")[1] + "." + str(get_index(node_name)) + "_" + relevant_letter
            layer = int(node_name.split(".")[1])

        # Handle attention hook_result
        elif "hook_result" in node_name or any([qkv_substring in node_name for qkv_substring in qkv_substrings]):
            name = "a" + node_name.split(".")[1] + "." + str(get_index(node_name))
            layer = int(node_name.split(".")[1])

        # Handle MLPs
        elif node_name.endswith("resid_mid"):
            raise ValueError("We removed resid_mid annotations. Call these mlp_in now.")
        elif "mlp" in node_name:
            name = "m" + node_name.split(".")[1]
            layer = int(node_name.split(".")[1])

        # Handle resid_post
        elif "resid_post" in node_name:
            name += "output"
            layer = 12

        # elif "mlp" in node_name:
        #     name += "m" + node_name.split(".")[1]
        else:
            raise ValueError(f"Unrecognized node name {node_name}")

    else:
        name = node_name
        # name = node_name + str(node.index.graphviz_index(use_actual_colon=True))

    # get layer by looking for number before first dot
    

    return layer, name