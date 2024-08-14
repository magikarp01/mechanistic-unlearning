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

def get_edge_mask_template(num_layers=12, num_heads=12, neox=False):
    """
    neox: MLP not after attn, instead they access same time, no edge between same layer attn and mlp
    """
    edge_mask_template = {}
    edge_mask_template["embed"] = torch.ones(size=(0,))
    num_prev_components = 1
    for layer in range(num_layers):
        for head in range(num_heads):
            edge_mask_template[f"a{layer}.{head}"] = torch.ones(size=(num_prev_components,))
        if neox:
            edge_mask_template[f"m{layer}"] = torch.ones(size=(num_prev_components,))
            num_prev_components += num_heads + 1
        else:
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


def get_masks_from_acdcpp_exp(acdcpp_exp, threshold=None):
    """
    acdcpp_exp is an instance of ACDCPPExperiment. threshold is a float, one of the thresholds specified in the thresholds list in acdcpp_exp. Formats the output into a nodes set, an edges set, an edge mask dict for edge masking/circuit breaking, and weight mask dicts for attn and mlp weight masking.
    """
    pruned_heads, num_passes, acdcpp_pruned_attrs, acdc_pruned_attrs, edges_after_acdcpp, edges_after_acdc = acdcpp_exp.run()

    acdcpp_edges = set()
    if threshold is None:
        threshold = list(edges_after_acdcpp.keys())[0]
    for edge in edges_after_acdcpp[threshold]:
        # split the edge into two nodes, e.g. blocks.1.attn.hook_result[:, :, 10]blocks.0.hook_mlp_in[:] into blocks.1.attn.hook_result[:, :, 10] and blocks.0.hook_mlp_in[:]
        node_1 = get_node_name(edge.split("]")[0]+"]", show_full_index=False)
        node_2 = get_node_name(edge.split("]")[1]+"]", show_full_index=False)
        if node_1 != node_2:
            acdcpp_edges.add((node_1, node_2))

    edge_mask_template = get_edge_mask_template()
    acdcpp_mask_dict = get_mask_from_edges(acdcpp_edges, edge_mask_template=edge_mask_template, num_layers=12, num_heads=12)
    acdcpp_nodes = get_nodes_from_edges(acdcpp_edges)
    acdcpp_weight_mask_attn_dict, acdcpp_weight_mask_mlp_dict = get_mask_components(acdcpp_nodes, num_layers=12, num_heads=12)

    return acdcpp_nodes, acdcpp_edges, acdcpp_mask_dict, acdcpp_weight_mask_attn_dict, acdcpp_weight_mask_mlp_dict

def get_formatted_edges_from_eap(eap_edges):
    converted_set = set()
    for edge in eap_edges:
        # Split the source and target node strings
        src, tgt, _ = edge
        
        # Convert format for source and extract layer
        if "mlp" in src:
            layer = int(src.split(".")[1])
            src = (layer, f"m{layer}")
        elif "head" in src:
            # print(f"{tgt.split('.')[1:3]}, {tgt}")
            layer, head = src.split(".")[1:3]
            src = (int(layer), f"a{layer}.{head}")
        
        # Convert format for target and extract layer
        if "mlp" in tgt:
            layer = int(tgt.split(".")[1])
            tgt = (layer, f"m{layer}")
        elif "head" in tgt:
            # print(f"{tgt.split('.')[1:3]}, {tgt}")
            layer, head = tgt.split(".")[1:3]
            tgt = (int(layer), f"a{layer}.{head}")
        
        # Add to set with reversed order
        converted_set.add((tgt, src))
    
    return converted_set


def get_masks_from_eap_exp(graph, threshold=0.001, filter_neox=False, **kwargs, ):
    """
    graph is an instance of EAPGraph. threshold is a float, one of the thresholds specified in the thresholds list in graph. Formats the output into a nodes set, an edges set, an edge mask dict for edge masking/circuit breaking, and weight mask dicts for attn and mlp weight masking.
    """
    eap_unformatted_edges = graph.top_edges(n=len(graph.eap_scores.flatten()), threshold=threshold)
    eap_edges = get_formatted_edges_from_eap(eap_unformatted_edges)
    if filter_neox:
        print(f"Old len: {len(eap_edges)}")
        new_eap_edges = set()
        for edge in eap_edges:
            # if first node is mlp layer l and second node is attn layer l, don't add edge
            if "m" in edge[0][1] and "a" in edge[1][1] and edge[0][0] == edge[1][0]:
                continue
            new_eap_edges.add(edge)
        eap_edges = new_eap_edges
        print(f"New len: {len(eap_edges)}")
        

    edge_mask_template = get_edge_mask_template(neox=filter_neox, **kwargs)
    eap_mask_dict = get_mask_from_edges(eap_edges, edge_mask_template=edge_mask_template, **kwargs)
    eap_nodes = get_nodes_from_edges(eap_edges)
    eap_weight_mask_attn_dict, eap_weight_mask_mlp_dict = get_mask_components(eap_nodes, **kwargs)

    return eap_nodes, eap_edges, eap_mask_dict, eap_weight_mask_attn_dict, eap_weight_mask_mlp_dict


def get_masks_from_ct_nodes(nodes_set, **kwargs):
    """
    Using a set of nodes that are important as determined by Causal Tracing, get the nodes, edges, and masks. Nodes should come in presorted.
    Assume node_set looks like {'a0.0', ..., 'm11'}. Want edges from later to earlier.

    """
    edges_set = set()
    """
    want: {((3, 'm3'), (3, 'a3.0')),
    ((4, 'm4'), (0, 'm0')),
    ((4, 'm4'), (3, 'a3.0')),
    ((5, 'a5.9'), (0, 'm0')),}
    """

    for first_node_idx in range(len(nodes_set)):
        if "a" in nodes_set[first_node_idx]:
            first_layer = int(nodes_set[first_node_idx].split(".")[0][1:])
        else:
            first_layer = int(nodes_set[first_node_idx][1:])

        for second_node_idx in range(first_node_idx):
            # get layer, allowing for either a or m. if a, format is a{l}.{h}, if m, format is m{l}, get l
            if "a" in nodes_set[second_node_idx]:
                second_layer = int(nodes_set[second_node_idx].split(".")[0][1:])
            else:
                second_layer = int(nodes_set[second_node_idx][1:])

            edges_set.add(((first_layer, nodes_set[first_node_idx]), (second_layer, nodes_set[second_node_idx])))
    
    
    edge_mask_template = get_edge_mask_template(**kwargs)
    ct_mask_dict = get_mask_from_edges(edges_set, edge_mask_template=edge_mask_template, **kwargs)
    ct_weight_mask_attn_dict, ct_weight_mask_mlp_dict = get_mask_components(nodes_set, **kwargs)

    return nodes_set, edges_set, ct_mask_dict, ct_weight_mask_attn_dict, ct_weight_mask_mlp_dict


import numpy as np
from collections import defaultdict

def convert_attrs_to_components(attrs, n_layers, n_heads, combine_heads=False):
    """
    attrs is dictionary of e.g. {'a0.0_q': float, 'm27_in': float}

    If combine_heads, then it will combine all 'a0.0_q', 'a0.1_q', ..., 'a0.15_q', etc into one component.
    """

    component_dict = defaultdict(int)
    attn_head_dict = defaultdict(dict)
    for layer in range(n_layers):
        for attn_type, component_name in [("q", f"blocks.{layer}.attn.hook_q"), ("k", f"blocks.{layer}.attn.hook_k"), ("v", f"blocks.{layer}.attn.hook_v"), ("result", f"blocks.{layer}.attn.hook_result")]:
            for head in range(n_heads):    
                if combine_heads:
                    component_dict[component_name] += attrs[f"a{layer}.{head}_{attn_type}"]
                else:
                    attn_head_dict[component_name][head] = attrs[f"a{layer}.{head}_{attn_type}"]
        for mlp_type, component_name in [("in", f"blocks.{layer}.mlp.hook_pre"), ("out", f"blocks.{layer}.mlp.hook_post")]:
            component_dict[component_name] += attrs[f"m{layer}_{mlp_type}"]
    if combine_heads:
        return (component_dict,)
    return (component_dict, attn_head_dict,)

def find_component_params(component_name, param_count_dict):
    for component_substr in param_count_dict.keys():
        if component_substr in component_name:
            return param_count_dict[component_substr]

def get_top_components(component_dict, attn_head_dict=None, n_heads=None, threshold=None, top_p=None, top_k=None, param_count=None, param_count_dict=None, use_abs=True):
    """
    component_dict is a dictionary of components to their importance values. If attn_head_dict is not None, then component_dict and attn_head_dict should not overlap in values.

    Can either use a threshold, top_p, or top_k to determine the top components to return (can only specify one). top_p should be a value ranging from 0 to 100. If use_abs is True, then it will take the absolute value of the importance values. 

    Returned final_components are all components (in the form of keys of tl_model.hook_dict), and returned final_attn_heads are a dictionary of components (keys are a subset of final_components that are attn components) to list of attention heads to mask over.
    """
    if attn_head_dict is not None:
        assert (component_dict.keys() & attn_head_dict.keys()) == set(), "Overlapping keys between component_dict and attn_head_dict"
    
    # assert only one of threshold, top_p, top_k is specified
    assert sum([threshold is not None, top_p is not None, top_k is not None, param_count is not None]) == 1, "Can only specify one of threshold, top_p, top_k, param_count"
    # will calculate a threshold for top_p or top_k

    if param_count is not None:
        assert param_count_dict is not None, "If param_count is specified, param_count_dict must also be specified"
        all_attr_values = list(component_dict.values())
        if attn_head_dict is not None:
            all_attr_values += [val for head_dict in attn_head_dict.values() for val in head_dict.values()]
        sorted_attrs = sorted(component_dict.items(), key=lambda x: np.abs(x[1]), reverse=True) if use_abs else sorted(component_dict.items(), key=lambda x: x[1], reverse=True)

        current_param_count = 0
        current_attr_idx = 0
        final_components = []
        final_attn_heads = defaultdict(list)
        while current_param_count < param_count:
            # find next most important component
            component, attr = sorted_attrs[current_attr_idx]
            final_components.append(component)
            current_param_count += find_component_params(component, param_count_dict)
            current_attr_idx += 1

            if "attn" in component:
                final_attn_heads[component] = list(range(n_heads))

    else:
        if top_p is not None:
            all_attr_values = list(component_dict.values())
            if attn_head_dict is not None:
                all_attr_values += [val for head_dict in attn_head_dict.values() for val in head_dict.values()]

            all_attr_values = np.array(all_attr_values)
            if use_abs:
                all_attr_values = np.abs(all_attr_values)
            print(f"{len(all_attr_values)=}")
            threshold = np.percentile(all_attr_values, 100 - top_p)
        elif top_k is not None:
            all_attr_values = list(component_dict.values())
            if attn_head_dict is not None:
                all_attr_values += [val for head_dict in attn_head_dict.values() for val in head_dict.values()]

            all_attr_values = np.array(all_attr_values)
            if use_abs:
                all_attr_values = np.abs(all_attr_values)
            threshold = np.sort(all_attr_values)[-top_k]
        
        print(f"Thresholding importance at {threshold}")
        final_components = []
        final_attn_heads = defaultdict(list)

        for component, importance in component_dict.items():
            if use_abs:
                importance = abs(importance)
            if importance >= threshold:
                final_components.append(component)    

        if attn_head_dict is not None:
            for component, head_dict in attn_head_dict.items():
                head_list = []
                for head, importance in head_dict.items():
                    if use_abs:
                        importance = abs(importance)
                    if importance >= threshold:
                        head_list.append(head)
                if len(head_list) > 0:
                    final_attn_heads[component] = head_list
                    final_components.append(component)
        else:
            for component in final_components:
                if "attn" in component:
                    # want to mask over all possible heads
                    final_attn_heads[component] = list(range(n_heads))
        
    return final_components, final_attn_heads

def get_component_name_from_ct(component, combine_heads, n_heads=None):
    final_components = []
    final_attn_heads = defaultdict(list)
    if combine_heads:
        layer = int(component[1:])
        # convert to component name
        if component[0] == "a":
            # add q, k, v, result to tunable params
            for attn_type in ['q', 'k', 'v', 'result']:
                final_components.append(f"blocks.{layer}.attn.hook_{attn_type}")
                final_attn_heads[f"blocks.{layer}.attn.hook_{attn_type}"] = list(range(n_heads))
        else:
            # add in, out to tunable params
            for mlp_type in ['pre', 'post']:
                final_components.append(f"blocks.{layer}.mlp.hook_{mlp_type}")
    else:
        layer = int(component.split(".")[0][1:])
        if component[0] == "a":
            head = int(component.split(".")[1])
            # add q, k, v, result to tunable params
            for attn_type in ['q', 'k', 'v', 'result']:
                final_components.append(f"blocks.{layer}.attn.hook_{attn_type}")
                final_attn_heads[f"blocks.{layer}.attn.hook_{attn_type}"].append(head)
        else:
            # add in, out to tunable params
            for mlp_type in ['pre', 'post']:
                final_components.append(f"blocks.{layer}.mlp.hook_{mlp_type}")

    return final_components, final_attn_heads

def get_top_components_no_subcomponents(attrs, n_layers, n_heads, threshold=None, top_p=None, top_k=None, param_count=None, param_count_dict=None, use_abs=True, combine_heads=False):
    combined_attrs = {}
    # if combine heads, then we will combine all heads into one component per layer
    if combine_heads:
        for layer in range(n_layers):
            combined_attrs[f"a{layer}"] = 0
            for head in range(n_heads):
                combined_attrs[f"a{layer}"] += attrs[f"a{layer}.{head}"]
            
            combined_attrs[f"m{layer}"] = attrs[f"m{layer}"]
    else:
        combined_attrs = attrs

    assert sum([threshold is not None, top_p is not None, top_k is not None, param_count is not None]) == 1, "Can only specify one of threshold, top_p, top_k, param_count"

    if param_count is not None:
        print(f"Using param_count")
        # calculate threshold which corresponds to param_count
        # use ordered dictionary to keep track of cumulative sum
        # iterate through ordered dictionary, subtracting values from param_count until param_count is less than or equal to 0
        # return the value (threshold) of the last element that was added

        sorted_attrs = sorted(combined_attrs.items(), key=lambda x: np.abs(x[1]), reverse=True)
        for component, importance in sorted_attrs:
            components, _ = get_component_name_from_ct(component, combine_heads, n_heads=n_heads)
            for component in components:
                param_count -= find_component_params(component, param_count_dict)
            print(component, param_count)
            if param_count <= 0:
                threshold = importance
                break

    elif top_p is not None:
        all_attr_values = list(combined_attrs.values())

        all_attr_values = np.array(all_attr_values)
        if use_abs:
            all_attr_values = np.abs(all_attr_values)
        print(f"{len(all_attr_values)=}")
        threshold = np.percentile(all_attr_values, 100 - top_p)

    elif top_k is not None:
        all_attr_values = list(combined_attrs.values())

        all_attr_values = np.array(all_attr_values)
        if use_abs:
            all_attr_values = np.abs(all_attr_values)
        threshold = np.sort(all_attr_values)[-top_k]
    
    print(f"Thresholding importance at {threshold}")

    final_components = set()
    final_attn_heads = defaultdict(list)

    for component, importance in combined_attrs.items():
        if use_abs:
            importance = abs(importance)
        if importance >= threshold:
            print(f"{component=}, {importance=} is being added")
            # if combine heads, then we will only have one component per layer
            components, attn_heads = get_component_name_from_ct(component, combine_heads, n_heads=n_heads)
            for component in components:
                final_components.add(component)

            for component, attn_heads in attn_heads.items():
                for attn_head in attn_heads:
                    final_attn_heads[component].append(attn_head)

            # if combine_heads:
            #     layer = int(component[1:])
            #     # convert to component name
            #     if component[0] == "a":
            #         # add q, k, v, result to tunable params
            #         for attn_type in ['q', 'k', 'v', 'result']:
            #             final_components.add(f"blocks.{layer}.attn.hook_{attn_type}")
            #             final_attn_heads[f"blocks.{layer}.attn.hook_{attn_type}"] = list(range(n_heads))
            #     else:
            #         # add in, out to tunable params
            #         for mlp_type in ['pre', 'post']:
            #             final_components.add(f"blocks.{layer}.mlp.hook_{mlp_type}")
            # else:
            #     layer = int(component.split(".")[0][1:])
            #     if component[0] == "a":
            #         head = int(component.split(".")[1])
            #         # add q, k, v, result to tunable params
            #         for attn_type in ['q', 'k', 'v', 'result']:
            #             final_components.add(f"blocks.{layer}.attn.hook_{attn_type}")
            #             final_attn_heads[f"blocks.{layer}.attn.hook_{attn_type}"].append(head)

            #     else:
            #         # add in, out to tunable params
            #         for mlp_type in ['pre', 'post']:
            #             final_components.add(f"blocks.{layer}.mlp.hook_{mlp_type}")
    return final_components, final_attn_heads


import math
import random
def get_random_components(n_layers, n_heads, top_p=None, top_k=None, combine_subcomponents=False, param_count=None, param_count_dict=None):
    assert sum([top_p is not None, top_k is not None, param_count is not None]) == 1, "Can only specify one of top_p, top_k, param_count"

    # select random subset of q, k, v, result, out, in. No combine_heads parameter because I don't think it makes sense to equally randomly select one 256-d head mask vs one 25k-d mlp mask
    if combine_subcomponents:
        possible_component_prefixes = []
        for layer in range(n_layers):
            possible_component_prefixes.append(f"blocks.{layer}.attn.hook")
            possible_component_prefixes.append(f"blocks.{layer}.mlp.hook")
        
        if param_count is not None:
            selected_component_prefixes = []
            while param_count > 0:
                component_prefix = random.choice(possible_component_prefixes)
                if "attn" in component_prefix:
                    for attn_type in ['q', 'k', 'v', 'result']:
                        param_count -= find_component_params(f"{component_prefix}_{attn_type}", param_count_dict)
                else:
                    for mlp_type in ['pre', 'post']:
                        param_count -= find_component_params(f"{component_prefix}_{mlp_type}", param_count_dict)

                # remove from possible_component_prefixes
                possible_component_prefixes.remove(component_prefix)
                selected_component_prefixes.append(component_prefix)
            
        else:
            if top_p is not None:
                top_k = int(math.ceil(top_p / 100 * len(possible_component_prefixes)))
            selected_component_prefixes = random.sample(possible_component_prefixes, top_k)

        final_components = []
        attn_dict = defaultdict(list)
        for component_prefix in selected_component_prefixes:
            if "attn" in component_prefix:
                for attn_type in ['q', 'k', 'v', 'result']:
                    final_components.append(f"{component_prefix}_{attn_type}")
                    attn_dict[f"{component_prefix}_{attn_type}"] = list(range(n_heads))
            else:
                for mlp_type in ['pre', 'post']:
                    final_components.append(f"{component_prefix}_{mlp_type}")
        
        return final_components, attn_dict
    else:
        all_possible_components = []
        for layer in range(n_layers):
            for attn_type in ['q', 'k', 'v', 'result']:
                all_possible_components.append(f"blocks.{layer}.attn.hook_{attn_type}")
            for mlp_type in ['pre', 'post']:
                all_possible_components.append(f"blocks.{layer}.mlp.hook_{mlp_type}")
        
        if param_count is not None:
            selected_components = []
            while param_count > 0:
                component = random.choice(all_possible_components)
                all_possible_components.remove(component)
                param_count -= find_component_params(component, param_count_dict)
                selected_components.append(component)
                print(component, param_count)
        elif top_p is not None:
            top_k = int(math.ceil(top_p / 100 * len(all_possible_components)))
            selected_components = random.sample(all_possible_components, top_k)
        
        attn_dict = defaultdict(list)
        for component in selected_components:
            if "attn" in component:
                attn_dict[component] = list(range(n_heads))
        
        return selected_components, attn_dict

def load_mask_from_state_dict(mask_path, n_heads):
    """
    For now only meant for random. DO NOT USE yet if combine_heads=False.
    """
    from circuit_breaking.src.masks import convert_param_name
    state_dict = torch.load(mask_path)
    final_components = []
    final_attn_heads = defaultdict(list)
    for key in state_dict.keys():
        if "masks.blocks" not in key:
            continue
        component_name = convert_param_name(key, inverse=True)
        # cut off "masks."
        component_name = component_name[6:]
        final_components.append(component_name)
        if "attn" in component_name:
            final_attn_heads[component_name] = list(range(n_heads))


def get_parameter(hf_model, component_name, model_type):
    if model_type == "gemma":
        layers_module = hf_model.model
    elif model_type == "pythia":
        layers_module = hf_model.gpt_neox

    layer_str, component_type, hook_type = component_name.split(".")[1:]
    layer = int(layer_str)

    if model_type == "gemma":
        if component_type == "attn":
            if hook_type == "hook_q":
                weight_param = layers_module.layers[layer].self_attn.q_proj.weight
            elif hook_type == "hook_k":
                weight_param = layers_module.layers[layer].self_attn.k_proj.weight
            elif hook_type == "hook_v":
                weight_param = layers_module.layers[layer].self_attn.v_proj.weight
            elif hook_type == "hook_result":
                # for now ignore, not sure if result maps to o_proj
                # print(f"Ignoring {component_name}")
                # weight_param = layers_module.layers[layer].self_attn.o_proj.weight
                weight_param = layers_module.layers[layer].self_attn.o_proj.weight
            else:
                print(f"Unknown component type {component_type}")
        elif component_type == "mlp":
            if hook_type == "hook_pre":
                weight_param = layers_module.layers[layer].mlp.up_proj.weight
            elif hook_type == "hook_post":
                weight_param = layers_module.layers[layer].mlp.down_proj.weight
            else:
                print(f"Unknown component type {component_type}")
        bias_param = None

    elif model_type == "pythia":
        if component_type == "attn":            
            weight_param = layers_module.layers[layer].attention.query_key_value.weight
            bias_param = layers_module.layers[layer].attention.query_key_value.bias
        elif component_type == "mlp":
            if hook_type == "hook_pre":
                weight_param = layers_module.layers[layer].mlp.dense_h_to_4h.weight
                bias_param = layers_module.layers[layer].mlp.dense_h_to_4h.bias
            elif hook_type == "hook_post":
                weight_param = layers_module.layers[layer].mlp.dense_4h_to_h.weight
                bias_param = layers_module.layers[layer].mlp.dense_4h_to_h.bias
            else:
                print(f"Unknown component type {component_type}")
    else:
        print(f"Unknown component type {component_type}")
    
    return (weight_param, bias_param)
    
def apply_localized_gradients(hf_model, components, model_type, verbose=False):
    # set everything else False
    for parameter in hf_model.parameters():
        parameter.requires_grad = False
    
    for component in components:
        if verbose:
            print(f"Setting {component} to True")
        params = get_parameter(hf_model, component, model_type)
        for i, param in enumerate(params):
            if param is None:
                if verbose:
                    print(f"Could not find {i}th parameter for {component}")
                continue
            param.requires_grad = True
