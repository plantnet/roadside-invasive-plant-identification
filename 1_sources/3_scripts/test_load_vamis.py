#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""This module provides basic code to load a vamis model"

Author: Vincent Espitalier <vincent.espitalier@cirad.fr>
        Hervé Goëau <herve.goeau@cirad.fr>
"""

#########################
# Appel au script
# $ conda activate pytorch2
# $ micromamba activate pytorch2
# $ python test_load_vamis.py


#########################
# Paramètres utilisateur

# Model: Beit-Danish        - détecteur 384x384
#f_resume_in   = ""                                                          
#model_input_size_xy_vamis = None

# Model: Beit-Danish vamisé - détecteur 768x1024 (height,width)
f_resume_in  = "../../0_datastore/5_models/model_vamis_state_dict_xp2.pth" 
model_input_size_xy_vamis = [768, 1017]


# The following parameters should not be changed, unless you know what you do
num_classes         = 7
model_name_timm     = "beit_base_patch16_384"
model_timm_infos    = {
    "model_input_size"  : 384,
    "patch_size"        : 16,
    "embedding_dim"     : 768,
    "channels"          : 3,
    "img_mean"          : [0.5, 0.5, 0.5],
    "img_std"           : [0.5, 0.5, 0.5]}

f_model_base         = "../../0_datastore/5_models/model_beit_danish.pth"          # Le Beit-Danish        - détecteur 384x384

#########################
# Chargement des librairies, définition des fonctions

import os
import torch
import timm
import json

def calc_nb_params_model(model):
    nb_params = sum(p.numel() for p in model.parameters())
    return nb_params


#########################
# Début du code principal


# Chargement du modèle
state_dict = None
if (len(f_model_base) > 2):
    if (not os.path.exists(f_model_base)):
        print("File doesn't exist: ", f_model_base)
        exit(-1)

    state_dict = torch.load(f_model_base, map_location="cpu")
    if 'state_dict_ema' in state_dict:
        state_dict = state_dict['state_dict_ema']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if ("head.bias" in state_dict.keys()):
        num_classes = state_dict["head.bias"].shape[0]

print("model_name_timm:",   model_name_timm)
print("model_input_size_xy_vamis:", model_input_size_xy_vamis)
print("Number of classes:", num_classes)
print("f_model_base:",       f_model_base)

if (state_dict != None):
    pretrained_cfg_overlay = dict(state_dict = state_dict)
else:
    pretrained_cfg_overlay = None

if (model_input_size_xy_vamis != None):
    # dim1 = Height (vertical)  ;  dim2 = Width (horizontal)
    args_model_kwargs = {'img_size': (model_input_size_xy_vamis[0], model_input_size_xy_vamis[1])}
    model_timm_infos["model_input_size"] = model_input_size_xy_vamis
else:
    args_model_kwargs = {}

#print(model_name_timm)
#print(num_classes)
#print(pretrained_cfg_overlay)
#print(args_model_kwargs)
model = timm.create_model(
    model_name  = model_name_timm,
    pretrained  = True,           # Si False -> Poids aléatoires
    num_classes = num_classes,
    #in_chans    = channels,
    #global_pool = None,
    #scriptable  = False,
    pretrained_cfg_overlay = pretrained_cfg_overlay,
    **args_model_kwargs)


if (state_dict != None):
    if (model_input_size_xy_vamis != None):
        # Patch for bug in timm:create_model(): It puts random weights in the head of the model.
        head_state_dict = {"weight" : state_dict["head.weight"], "bias" : state_dict["head.bias"]}
        msg = model.head.load_state_dict(head_state_dict)
        print('Pretrained head weights found at {} and loaded with msg: {}'.format(f_model_base, msg))
    else:
        msg = model.load_state_dict(state_dict, strict = True) #False)
        print('Pretrained weights found at {} and loaded with msg: {}'.format(f_model_base, msg))
        

# Chargement du resume
if (len(f_resume_in) > 2):
    if (not os.path.exists(f_resume_in)):
        print("File doesn't exist: ", f_resume_in)
        exit(-1)

    state_dict = torch.load(f_resume_in)
    if 'state_dict_ema' in state_dict:
        state_dict = state_dict['state_dict_ema']
    elif 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if ("head.bias" in state_dict.keys()):
        num_classes = state_dict["head.bias"].shape[0]
        
    model.load_state_dict(state_dict, strict = True)

# Output dict:
# 1) Add timm infos
model_infos = model_timm_infos.copy()

# 2) Add local infos
model_infos["num_classes"]     = num_classes
model_infos["model_name_timm"] = model_name_timm
model_infos["f_model_base"]     = f_model_base
model_infos["f_resume_in"]     = f_resume_in
model_infos["model_input_size_xy_vamis"]  = model_input_size_xy_vamis    # VaMIS



####################
# Utilisation du modèle

# A ce stade:
# > La variable 'model' contient le modèle en question
# > Le dictionnaire 'model_infos' contient toutes les infos pertinentes relatives au modèle
#   (nb classes, model input size etc.)

print("Modèle:", model)

print("model_infos:", json.dumps(model_infos, indent = 2))

print("Nombre de paramètres du modèle: ", calc_nb_params_model(model))
