#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""This module provides code to manage the deep learning models: Load them, customize them for research purpose

Author: Vincent Espitalier <vincent.espitalier@cirad.fr>
        Hervé Goëau <herve.goeau@cirad.fr>
"""

import os
import math
import torch
from torch import nn

import timm
from timm import create_model as timm_create_model

def calc_nb_params_model(model):
    nb_params = sum(p.numel() for p in model.parameters())
    return nb_params

def calc_nb_learnable_params_model(model):
    nb_params = sum(p.numel() if (p.requires_grad == True) else 0 for p in model.parameters())
    return nb_params


def load_model_timm(model_name, models_infos, model_input_size_xy_vamis, d_models_folder, device):
    print("Call to lib_models_timm > load_model_timm()")

    if (not ("local_infos" in models_infos.keys())):
        print("Error in 'models_infos' dict: Missing key (1): local_infos")
        exit(-1)

    if (not (model_name in models_infos["local_infos"].keys())):
        print("Error in 'models_infos' dict: Missing key (2): ", model_name)
        exit(-1)
    
    model_local_infos = models_infos["local_infos"][model_name]
    model_name_timm = model_local_infos["model_name_timm"]
    num_classes     = model_local_infos["num_classes"]
    f_resume_in     = model_local_infos["f_resume_in"]

    if (not (model_name_timm in models_infos["timm_infos"].keys())):
        print("The timm model is not referenced in the models_infos['timm_infos']:", model_name_timm)
        exit(-1)
    
    model_timm_infos = models_infos["timm_infos"][model_name_timm]

    state_dict = None
    if (len(f_resume_in) > 2):
        f_resume_in_full = os.path.join(d_models_folder, f_resume_in)
        if (not os.path.exists(f_resume_in_full)):
            print("File doesn't exist: ", f_resume_in_full)
            exit(-1)

        state_dict = torch.load(f_resume_in_full, map_location="cpu")
        if 'state_dict_ema' in state_dict:
            state_dict = state_dict['state_dict_ema']
        elif 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        #state_dict = torch.load(extractor_model_path, map_location="cpu")['state_dict']

        if ("head.bias" in state_dict.keys()):
            num_classes = state_dict["head.bias"].shape[0]


    print("model_name:",        model_name)
    print("model_name_timm:",   model_name_timm)
    print("model_input_size_xy_vamis:", model_input_size_xy_vamis)
    print("Number of classes:", num_classes)
    print("f_resume_in:",       f_resume_in)


    #channels = model_timm_infos["channels"]
    if (state_dict != None):
        pretrained_cfg_overlay = dict(state_dict = state_dict)
    else:
        pretrained_cfg_overlay = None

    if (model_input_size_xy_vamis != None):
        # dim1 = Height (vertical)  ;  dim2 = Width (horizontal)
        args_model_kwargs = {'img_size': (model_input_size_xy_vamis[0], model_input_size_xy_vamis[1])}
    else:
        args_model_kwargs = {}

    try:
        print(model_name_timm)
        print(num_classes)
        #print(pretrained_cfg_overlay)
        print(args_model_kwargs)
        model = timm_create_model(
            model_name  = model_name_timm,
            pretrained  = True,           # Si False -> Poids aléatoires
            num_classes = num_classes,
            #in_chans    = channels,
            #global_pool = None,
            #scriptable  = False,
            pretrained_cfg_overlay = pretrained_cfg_overlay,
            **args_model_kwargs)

    except:
        print("Error in load_model_timm(), with model_name_timm = ", model_name_timm)
        print("Is timm version recent enough ? For instance, DINOv2 registers requires at least timm 0.9.9")
        print("Current timm version:", timm.__version__)
        exit(-1)

    if (state_dict != None):
        if (model_input_size_xy_vamis != None):
            # Patch for bug in timm:create_model(): It puts random weights in the head of the model.
            head_state_dict = {"weight" : state_dict["head.weight"], "bias" : state_dict["head.bias"]}
            msg = model.head.load_state_dict(head_state_dict)
            print('Pretrained head weights found at {} and loaded with msg: {}'.format(f_resume_in_full, msg))
        else:
            msg = model.load_state_dict(state_dict, strict = True) #False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(f_resume_in_full, msg))
    #print('Pretrained weights found at {} and loaded with msg: {}'.format(f_resume_in_full, None))

    model.eval()
    model = model.to(device)


    # Output dict:
    # 1) Add timm infos
    model_infos = model_timm_infos.copy()

    # 2) Add local infos
    model_infos["num_classes"]     = num_classes
    model_infos["model_name"]      = model_name
    model_infos["model_name_timm"] = model_name_timm
    model_infos["f_resume_in"]     = f_resume_in
    model_infos["model_input_size_xy_vamis"]  = model_input_size_xy_vamis    # VaMIS
    #exit()
    return [model, model_infos]    



class transpose_layer(nn.Module):
    def __init__(self, dimA, dimB):
        super().__init__()
        self.dimA = dimA
        self.dimB = dimB
    def forward(self, x):
        return x.transpose(self.dimA, self.dimB)

class squeeze_layer(nn.Module):
    def __init__(self, dimA):
        super().__init__()
        self.dimA = dimA   
    def forward(self, x):
        return x.squeeze(self.dimA)

class softmax_layer(nn.Module):
    def __init__(self, dimA):
        super().__init__()
        self.dimA = dimA   
    def forward(self, x):
        return nn.functional.softmax(input = x, dim = self.dimA)

class shrink_layer(nn.Module):
    def __init__(self, dimA, dimB):
        super().__init__()
        self.dimA = dimA   
        self.dimB = dimB
    def forward(self, x):
        return (x.transpose(0, self.dimA)[0:self.dimB]).transpose(0, self.dimA)

class max_layer(nn.Module):
    def __init__(self, dimA):
        super().__init__()
        self.dimA = dimA   
    def forward(self, x):
        return torch.max(input = x, dim = self.dimA).values

class argmax_layer(nn.Module):
    def __init__(self, dimA):
        super().__init__()
        self.dimA = dimA   
    def forward(self, x):
        return torch.max(input = x, dim = self.dimA).indices
        
class diaglinear_layer(nn.Module):
    def __init__(self, dimA):
        super().__init__()
        self.dimA = dimA   
        self.weight = torch.ones(dimA, device = torch.device("cuda"))
    def forward(self, x):
        return torch.matmul(x, torch.diag(self.weight))


def create_model_weighted_sum_or_max_aggregation(
        n_views,
        n_model_views,
        n_features,
        n_species,
        p_dropout,
        bool_max_logits = False,
        bool_argmax = False,
        bool_layer_norm = False,
        bool_max_add_softmax_layer = False,
        bool_max_linear_avant_max = False,
        bool_final_sigmoid = False,
        init_classif_linear = None):

    # Création du modèle, comme succession de couches
    model = nn.Sequential()

    if (bool_layer_norm):
        # Layer norm sans paramètre d'apprentissage (sigma = 1 / sqrt(768))
        #dim_layer_norm = 10
        dim_layer_norm = n_features
        sigma = 1. / math.sqrt(dim_layer_norm)
        model.add_module("layer_norm",     nn.LayerNorm(normalized_shape = [dim_layer_norm], elementwise_affine = True))
        model.layer_norm.weight = nn.Parameter(torch.full(size = [dim_layer_norm], fill_value = sigma), requires_grad=False)
        for param in model.layer_norm.parameters():
            param.requires_grad = False
        #print(model.layer_norm.weight.shape)
        #print(list(model.layer_norm.parameters()))
        #print(model.layer_norm.state_dict())
        #exit()

    model.add_module("dropout1",           nn.Dropout(p_dropout))
    model.add_module("linear_classif",     nn.Linear(in_features = n_features, out_features = n_species, bias = True))
    model.add_module("transpose",          transpose_layer(1, 2))

    if (bool_max_add_softmax_layer):
        model.add_module("softmax",        softmax_layer(1))

    if (n_model_views < n_views):
        model.add_module("shrink_layer",   shrink_layer(2, n_model_views))

    if (not (bool_max_logits)):
        model.add_module("dropout2",           nn.Dropout(p_dropout))
        model.add_module("linear_weightedsum", nn.Linear(in_features = n_model_views, out_features = 1, bias = False))
        model.add_module("squeeze",            squeeze_layer(2))
    else:
        if (bool_max_linear_avant_max):
            model.add_module("linear_avant_max",   diaglinear_layer(n_model_views))
        if (bool_argmax):
            model.add_module("argmax",             argmax_layer(2))
        else:
            model.add_module("max",                max_layer(2))
        

    if (bool_final_sigmoid):
        model.add_module("sigmoid",                nn.Sigmoid())

    # Initialisation de ses poids
    if (init_classif_linear != None):
        init_classif_linear_state_dict = init_classif_linear.state_dict()
        model.linear_classif.weight   = nn.Parameter(init_classif_linear_state_dict["weight"])
        model.linear_classif.bias     = nn.Parameter(init_classif_linear_state_dict["bias"])
    else:
        nn.init.xavier_uniform_(model.linear_classif.weight)
        model.linear_classif.bias.data.fill_(0.0)

    if (not (bool_max_logits)):
        model.linear_weightedsum.weight.data.fill_(1. / n_views)
    else:
        if (bool_max_linear_avant_max):
            # Cas du Max
            # On rajoute une pondération des 926 vues, qui s'implémente/se matérialise ici
            # par une couche linear avec une matrice de poids qui est diagonale.
            #model.linear_avant_max.weight = nn.Parameter(torch.diag(torch.diag(model.linear_avant_max.weight)))
            #model.linear_avant_max.weight = nn.Parameter(model.linear_avant_max.weight)
            print("create_model_weighted_sum_or_max_aggregation : Désactivé: model.linear_avant_max.weight = nn.Parameter(model.linear_avant_max.weight)")
            exit(-1)

    print(model)
    print("Nombre de paramètres:", calc_nb_params_model(model))

    if (False):
        filename_model_weights_out = "patch_output_model_statedict.pth"
        torch.save(model.state_dict(), filename_model_weights_out)
        print("Patch pour sauvegarder le state_dict du modèle")
        print("Fichier state_dict écrit:", filename_model_weights_out)
        exit() 

    model_infos = {}
    model_infos["model_name"]      = "tiling"
    return [model, model_infos]



class flatten_layer(nn.Module):
    def __init__(self, dimA):
        super().__init__()
        self.dimA = dimA
    def forward(self, x):
        return torch.flatten(x, start_dim = self.dimA)


def create_model_fully_connected(
        n_views,
        n_model_views,
        n_features,
        n_species,
        p_dropout):

    # Création du modèle, comme succession de couches
    model = nn.Sequential()
    if (n_model_views < n_views):
        model.add_module("shrink_layer",   shrink_layer(1, n_model_views))
   
    model.add_module("flatten_layer",      flatten_layer(1))
    model.add_module("dropout",            nn.Dropout(p_dropout))
    model.add_module("linear_classif",     nn.Linear(in_features = n_model_views * n_features, out_features = n_species, bias = True))

    print(model)
    print("Nombre de paramètres:", calc_nb_params_model(model))

    model_infos = {}
    model_infos["model_name"]      = "tiling"
    print("debug1640 fc")
    return [model, model_infos]

def load_aggregation_model(
        aggregation_type,
        n_views,
        n_model_views,
        n_features,
        n_species,
        p_dropout,
        bool_argmax                 = False,
        bool_layer_norm             = False,
        bool_max_add_softmax_layer  = False,
        bool_max_linear_avant_max   = False,
        bool_final_sigmoid          = False,
        bool_attention_feed_forward = False,
        init_classif_linear = None):

    print(" ")
    print("Appel à load_aggregation_model, avec paramètres:")
    print("aggregation_type:", aggregation_type)
    print("n_views:", n_views)
    print("n_model_views:", n_model_views)
    print("n_features:", n_features)
    print("n_species:", n_species)
    print("p_dropout:", p_dropout)
    print("bool_argmax", bool_argmax)
    print("bool_layer_norm:", bool_layer_norm)
    print("bool_max_add_softmax_layer:", bool_max_add_softmax_layer)
    print("bool_max_linear_avant_max:", bool_max_linear_avant_max)
    print("bool_final_sigmoid:", bool_final_sigmoid)
    print("bool_attention_feed_forward:", bool_attention_feed_forward)
    print("init_classif_linear:")
    print(init_classif_linear)
    print(" ")
    

    if (aggregation_type == "max"):
        return create_model_weighted_sum_or_max_aggregation(
            n_views         = n_views,
            n_model_views   = n_model_views,
            n_features      = n_features,
            n_species       = n_species,
            p_dropout       = p_dropout,
            bool_max_logits = True,
            bool_argmax     = bool_argmax,
            bool_layer_norm = bool_layer_norm,
            bool_max_add_softmax_layer = bool_max_add_softmax_layer,
            bool_max_linear_avant_max  = bool_max_linear_avant_max,
            bool_final_sigmoid         = bool_final_sigmoid,
            init_classif_linear        = init_classif_linear)

    if (aggregation_type == "weighted_sum"):
        return create_model_weighted_sum_or_max_aggregation(
            n_views         = n_views,
            n_model_views   = n_model_views,
            n_features      = n_features,
            n_species       = n_species,
            p_dropout       = p_dropout,
            bool_max_logits = False,
            bool_argmax     = False,
            bool_layer_norm = bool_layer_norm,
            bool_max_add_softmax_layer = bool_max_add_softmax_layer,
            bool_max_linear_avant_max  = bool_max_linear_avant_max,
            bool_final_sigmoid         = bool_final_sigmoid,
            init_classif_linear        = init_classif_linear)

    if (aggregation_type == "fully_connected"):
        return create_model_fully_connected(
            n_views         = n_views,
            n_model_views   = n_model_views,
            n_features      = n_features,
            n_species       = n_species,
            p_dropout       = p_dropout)        


    if (aggregation_type == "attention"):
        print("Erreur: load_aggregation_model : aggregation_type 'attention' pas implémenté.")
        exit(-1)
        return None

    print("Erreur: load_aggregation_model: aggregation_type non reconnu:", aggregation_type)
    exit(-1)
        
    return None


# Centrer réduire les poids d'un modèle et enlever le bias
def model_centrer_reduire_sans_biais(f_model_in, f_model_out):
    dict_model = torch.load(f_model_in)

    print("Appel à centrer_reduire_sans_biais_modele")
    print("f_model_in:", f_model_in)
    print("f_model_out:", f_model_out)

    n_species, n_features = list(dict_model["linear_classif.weight"].shape)
    nouvelle_stdev = 1. / math.sqrt(768)

    means = torch.mean(dict_model["linear_classif.weight"], 1)
    stds  = torch.std(dict_model["linear_classif.weight"], 1)
    biases = dict_model["linear_classif.bias"]
    print("Moments avant:", means, stds, biases)

    for specie_index in range(n_species):
        # Attention: Change la qualité du classifier si les deep features ne sont pas centrées par la layer norm
        dict_model["linear_classif.weight"][specie_index] = (dict_model["linear_classif.weight"][specie_index] - means[specie_index])

        # Ne change pas la qualité du classifier
        dict_model["linear_classif.weight"][specie_index] = dict_model["linear_classif.weight"][specie_index] / stds[specie_index] * nouvelle_stdev 
        dict_model["linear_classif.bias"][specie_index] = 0.0
        
    means = torch.mean(dict_model["linear_classif.weight"], 1)
    stds = torch.std(dict_model["linear_classif.weight"], 1)
    biases = dict_model["linear_classif.bias"]
    print("Moments après:", means, stds, biases)

    torch.save(dict_model, f_model_out)
    print("Fichier écrit:", f_model_out)
