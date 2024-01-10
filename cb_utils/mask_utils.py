from torch.optim import AdamW
import torch
import pickle
import datasets
from tqdm import tqdm_notebook as tqdm
from itertools import cycle
# from eval import evaluate_model
import plotly.express as px

def get_default_gpt2_param_names(num_layers=12):
    """
    Get default param names for GPT2 model with num_layers layers.
    """
    """default_param_names = ['output_mask',
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
    'blocks.11.edge_mask_mlp']"""
    default_param_names = ['output_mask']
    for i in range(num_layers):
        default_param_names.append(f'blocks.{i}.edge_mask_attentions')
        default_param_names.append(f'blocks.{i}.edge_mask_mlp')
    return default_param_names

# def get_default_edge_mask_dict(num_layers=12, num_heads=12):



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


def get_all_possible_nodes(mask_params=None, param_names=None, num_layers=12, num_heads=12, ):
    """
    Get node names in format of 
    [(-1, 'embed'),
    (0, 'a0.0'),
    (0, 'a0.1'),
    (0, 'a0.2'),
    (0, 'a0.3'),
    (0, 'a0.4'),
    (0, 'a0.5'),
    (0, 'm0'),
    ...,
    (12, 'output')]
    If mask_params and param_names are provided, then will use that + num_heads. If not, will use default param names for num_layers and num_heads.
    """
    if mask_params is not None and param_names is not None:
        # associate each node with a position
        all_possible_nodes = [(-1, "embed")]
        mask_dict = {}
        # empty tensor
        mask_dict["embed"] = torch.zeros(size=(0,))
        for idx in range(len(mask_params)):
            if "attention" in param_names[idx]:
                layer = int(param_names[idx].split(".")[1])
                for i in range(num_heads):
                    all_possible_nodes.append((layer, f"a{layer}.{i}"))
                    mask_dict[f"a{layer}.{i}"] = mask_params[idx][:,i].detach().cpu()
            elif "mlp" in param_names[idx]:
                layer = int(param_names[idx].split(".")[1])
                all_possible_nodes.append((layer, f"m{layer}"))
                mask_dict[f"m{layer}"] = mask_params[idx].detach().cpu()
        all_possible_nodes.append((num_layers, "output"))
    else:
        all_possible_nodes = [(-1, "embed")]
        for layer in range(num_layers):
            for head in range(num_heads):
                all_possible_nodes.append((layer, f"a{layer}.{head}"))
            all_possible_nodes.append((layer, f"m{layer}"))
        all_possible_nodes.append((num_layers, "output"))
    return all_possible_nodes


def get_nodes_and_edges(mask_params, param_names=None, edge_0=True):
    """
    If edge_0 is True, then edges are between nodes with mask value 0. Else, edges are between nodes with mask value 1.
    Returns all_possible_nodes, nodes_with_edges, edges, mask_dict
    """
    if param_names is None:
        param_names = get_default_gpt2_param_names()

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
    """Turn acdcpp node name into the circuit breaking node name. Flattens q, k, v of attention heads all into just attn."""

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

def get_edge_mask_template(num_layers=12, num_heads=12):
    edge_mask_template = {}
    edge_mask_template["embed"] = torch.ones(size=(0,))
    num_prev_components = 1
    for layer in range(num_layers):
        for head in range(num_heads):
            edge_mask_template[f"a{layer}.{head}"] = torch.ones(size=(num_prev_components,))
        num_prev_components += num_heads
        edge_mask_template[f"m{layer}"] = torch.ones(size=(num_prev_components,))
        num_prev_components += 1
    edge_mask_template["output"] = torch.ones(size=(num_prev_components,))
    return edge_mask_template

# Convert edges back to edge mask
def get_mask_from_edges(edges, edge_mask_template=None, all_possible_nodes=None, edge_0=True, num_layers=12, num_heads=12):
    """
    edges is a set of e.g. {((0, 'a0.1'), (-1, 'embed')),
        ((0, 'a0.10'), (-1, 'embed')),
        ((2, 'a2.2'), (-1, 'embed')),
        ((2, 'a2.2'), (0, 'a0.1')),}
        
    edge_mask_template: Doesn't matter what values, just needs shape of 
        {'embed': tensor([]),
        'a0.0': tensor([0.]),
        'a0.1': tensor([1.]),
        'a0.2': tensor([0.]),
        'a0.3': tensor([1.]),
        'a0.4': tensor([0.]),
        'a0.5': tensor([1.]),
        'm0': tensor([1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.]),
        ...}

    all_possible_nodes needs shape of
        [(-1, 'embed'),
        (0, 'a0.0'),
        (0, 'a0.1'),
        (0, 'a0.2'),
        (0, 'a0.3'),
        (0, 'a0.4'),
        (0, 'a0.5'),
        (0, 'm0'),
        ...,
        (12, 'output')]

    Outputs an edge_mask with actual edges with same form of weight_mask_template
    """
    if edge_mask_template is None:
        edge_mask_template = get_edge_mask_template(num_layers=num_layers, num_heads=num_heads)

    if all_possible_nodes is None:
        all_possible_nodes = get_all_possible_nodes(num_layers=num_layers, num_heads=num_heads)
    
    new_mask_dict = {}
    for node_name in edge_mask_template:
        new_mask_dict[node_name] = torch.ones_like(edge_mask_template[node_name]) if edge_0 else torch.zeros_like(edge_mask_template[node_name])
    
    node_indices = {node_name: idx for idx, node_name in enumerate(all_possible_nodes)}
    for edge in edges:
        try:
            new_mask_dict[edge[0][1]][node_indices[edge[1]]] = 0 if edge_0 else 1
        except:
            continue
    return new_mask_dict

def convert_mask_dict_to_params(mask_dict):
    """
    Takes a mask dict in format of 
        {'embed': tensor([]),
        'a0.0': tensor([0.]),
        'a0.1': tensor([1.]),
        'a0.2': tensor([0.]),
        'a0.3': tensor([1.]),
        'a0.4': tensor([0.]),
        'a0.5': tensor([1.]),
        'm0': tensor([1., 0., 1., 0., 0., 1., 0., 0., 0., 0., 0., 1., 0.]),
        ...}
    Returns mask params (that can be used in model state dict, look like mask_params, can be used in get_nodes_and_edges()).
    """
    mask_params = []
    # first output_mask
    mask_params.append(mask_dict["output"])
    for layer in range(12):
        attn_tensors = []
        for head in range(12):
            attn_tensors.append(mask_dict[f"a{layer}.{head}"])
        mask_params.append(torch.stack(attn_tensors, dim=1))
        mask_params.append(mask_dict[f"m{layer}"])
    return mask_params

def get_nodes_from_edges(edges_set):
    """
    edges_set looks like {((0, 'a0.10'), (-1, 'embed')),
    ((0, 'a0.9'), (-1, 'embed')),
    ((0, 'm0'), (-1, 'embed'))}
    Get all nodes in edges_set.
    """
    nodes = set()
    for edge in edges_set:
        for node in edge:
            nodes.add(node)
    return nodes

def get_mask_components(nodes_set, num_layers=12, num_heads=12):
    """
    nodes_set looks like {(-1, 'embed'),
    (0, 'a0.0'),
    (0, 'a0.1'),
    (0, 'a0.2'),
    (0, 'a0.3'),
    (0, 'a0.4'),
    (0, 'a0.5'),
    (0, 'm0'),
    ...,
    (12, 'output')}
    Prepare the inputs to weight_mask_attn_dict and weight_mask_mlp_dict. weight_mask_attn_dict should be a dictionary, mapping layer number to list of attention heads that are part of the nodes set. weight_mask_mlp_dict should be a dictionary, mapping layer number to bool, True if the MLP is part of the nodes set.
    """
    weight_mask_attn_dict = {}
    weight_mask_mlp_dict = {}
    for layer in range(num_layers):
        weight_mask_attn_dict[layer] = []
        for head in range(num_heads):
            if (layer, f"a{layer}.{head}") in nodes_set:
                weight_mask_attn_dict[layer].append(head)
        weight_mask_mlp_dict[layer] = (layer, f"m{layer}") in nodes_set
    return weight_mask_attn_dict, weight_mask_mlp_dict
