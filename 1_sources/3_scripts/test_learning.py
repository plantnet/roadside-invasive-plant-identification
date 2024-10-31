#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""This module provides code to test the learning functions of the pipeline (for tiling)"

Author: Vincent Espitalier <vincent.espitalier@cirad.fr>
        Hervé Goëau <herve.goeau@cirad.fr>
"""

#######################
## Commandes de lancement du script
## $ conda activate pytorch2
## $ micromamba activate pytorch2
## $ python test_learning.py
## $ python test_learning.py >> test_learning.log
## $ watch "tail -n 20 test_learning.log"
## $ less test_learning.log

## Commandes Tensorboard
# $ pip install tensorboard         # Installation de tensorboard (prend quelques secondes)
# $ tensorboard --logdir=runs       # Lancement de tensorboard


#bool_perform_learning = False
bool_perform_learning = True

#######################
## Paramètres utilisateur

bool_use_BCE_loss = True
#bool_use_BCE_loss = False

#bool_use_tensorboard = True
bool_use_tensorboard = False

#init_learning_rate = 0.003
#init_learning_rate = 0.01  # test - df norm
init_learning_rate = 0.02  # test - df norm
#init_learning_rate = 0.001  # test - optimal pour les df pas norm - jeudi 15/02 xp5h
#init_learning_rate = 0.0005  # test
#init_learning_rate = 0.0001  # XP5h 
#init_learning_rate = 0.00005  # 
#init_learning_rate = 0.00003  # test
#init_learning_rate = 0.00001  # test
#init_learning_rate = 0.0000001  # XP5g
#init_learning_rate = 0.000000001  # dummy pour tests

#weight_decay = 0       # val_loss: 0.41195
weight_decay = 0.001
#weight_decay = 0.002
#weight_decay = 0.01
#weight_decay = 0.05
#weight_decay = 0.1
#weight_decay = 0.2

#p_dropout = 0.0
p_dropout = 0.01
#p_dropout = 0.02
#p_dropout = 0.03
#p_dropout = 0.04
#p_dropout = 0.05
#p_dropout = 0.10
#p_dropout = 0.25
#p_dropout = 0.35
#p_dropout = 0.5
#p_dropout = 0.8     # XP5d, XP5e
#p_dropout = 0.9    # Valeur optimale apparemment (XP5b) - Permet de limiter en partie l'overfitting
#p_dropout = 0.95   # xp5g

label_smoothing_factor = -1.0   # Désactivé
#label_smoothing_factor = 0.0
#label_smoothing_factor = 0.1   # Toutes XPs
#label_smoothing_factor = 0.2

#batch_size = 4     # Pour debug
#batch_size = 64   # 11-13s / epoch avec 1,2,3 workers - pas mal mais un peu instable
#batch_size = 128  # 11-13s / epoch avec 1,2,3 workers
batch_size = 256   # 12s  - meilleur pour l'instant
 
#num_workers = 1
num_workers = 2    #   (meilleur choix)
#num_workers = 3
#num_workers = 4

#n_epochs = 0
#n_epochs = 1
#n_epochs = 5
n_epochs = 1000

early_stop_patience = 7
early_stop_min_delta = 0.00001
reduce_lr_patience = 3
reduce_lr_factor = 0.3

# Paramètres de structure du modèle
# Layer norm sans paramètre d'apprentissage (sigma = 1 / sqrt(768))
#bool_layer_norm = False
bool_layer_norm = True

bool_add_softmax_layer = False
#bool_add_softmax_layer = True  # N'aide pas

bool_max_logits = True    # XP4, XP5c, XP5d
#bool_max_logits = False   # XP5a, XP5b

# On rajoute une pondération des 926 vues, qui s'implémente/se matérialise ici
# par une couche linear avec une matrice de poids qui est diagonale.
#bool_linear_avant_max = True  # N'aide pas
bool_linear_avant_max = False

n_model_views = 926     # XP5a,b,c,d,g
#n_model_views = 14     # XP5e, XP5f - Utiliser les 2 premieres échelles seulement soit 2+12=14vues (résolution VAMIS SVGA 1017x768).


f_resume_in   = ""   # Resume
#f_resume_in   = "xp5c_patch_output_model_statedict.pth"   # xp5c utilisé pour eval dans le pipeline gen
#f_resume_in   = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__nepochs107_val_loss0.4503.pth"
#f_resume_in   = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__wd10_lr_0.003_nepochs16_val_loss0.4273_overfit.pth"
#f_resume_in   = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__wd10_nepochs29_val_loss0.4292.pth"

# Ensemble model
#f_resume_in = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__nepochs107_val_loss0.4503.pth"      # ens-model-1
#f_resume_in = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__nepochs19_val_loss0.4455.pth"       # ens-model-2
#f_resume_in = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__wd10_nepochs29_val_loss0.4292.pth"  # ens-model-3
#f_resume_in = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__wd1.2_nepochs23_val_loss0.4393.pth" # ens-model-4
#f_resume_in = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__wd15_nepochs18_val_loss0.4320.pth"  # ens-model-5
#f_resume_in = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__wd20_nepochs23_val_loss0.4249.pth"  # ens-model-6
#f_resume_in = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__wd30_nepochs18_val_loss0.4381.pth"  # ens-model-7
#f_resume_in = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__wd25_nepochs18_val_loss0.4322.pth"  # ens-model-8

#f_resume_in  = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__pickup_model_val_loss0.4151.pth"  # pickup model
#f_resume_in  = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__pickup_model_avec_rosa.pth"  # pickup model
#f_resume_in  = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__pickup_model_avec_rosadiv10.pth" #

#f_resume_in  = "trained_model/cp_xp5h_max_logits_BCE_sans_no_species__pickup_model_mercredi_soir_tiling_FT_val_loss0.4090.pth" #


#f_resume_in  = "cp_xp5h_BCE_LN__pickup_strategy.pth" # pickup samedi (17-02)
#f_resume_in  = "model_state_dict_xp5h_BCE_LN__pickup_strategy.pth" # pickup samedi (17-02)

# Reproduire l'xp5c
#f_resume_in  = "verif_model_head_xp5c/model_best_head_sans_heracleum_verif_2024.pth"   # NE PAS UTILISER
#f_resume_in  = "verif_model_head_xp5c/model_best_head_sans_heracleum_div10_verif_2024.pth"  # <-  Bon modèle
#f_resume_in  = "verif_model_head_xp5c/model_best_head_sans_heracleum_div10_verif_2024_avec_layernorm.pth"  # avec LN


#tag_tiling = "xp5h_max_logits_BCE_sans_no_species__pickup_model_val_loss0.4151"
#tag_tiling = "xp5h_max_logits_BCE_sans_no_species__pickup_model_avec_rosa"
#tag_tiling = "xp5h_max_logits_BCE_sans_no_species__pickup_model_avec_rosadiv10"  # **final xp5h**
#tag_tiling = "xp5c_reproduce_2024_div10_verif_2024_val_predictions"

#tag_tiling = "xp5h_max_logits_BCE_sans_no_species_ensemble-model-8"

#tag_tiling = "xp5h_max_logits_BCE_sans_no_species__wd10_lr_0.003_nepochs16_val_loss0.4273_overfit"  # Inférence seule
#tag_tiling = "xp5h_max_logits_BCE_sans_no_species__wd10_nepochs29_val_loss0.4292"  # Inférence seule

#tag_tiling = "xp5h_max_logits_BCE_sans_no_species__wd" + str(weight_decay) + "_depuis_xp5c"  # Inférence seule
#tag_tiling = "cp_xp5h_max_logits_BCE_sans_no_species__pickup_model_mercredi_soir_tiling_FT_val_loss0.4090"  # Inférence seule

#tag_tiling = "xp5h_max_logits_BCE_sans_no_species_dfcentrereduit__wd" + str(weight_decay) + "_do" + str(p_dropout)

xp_param_string = "wd=" + str(weight_decay) + "_do=" + str(p_dropout) + "_lr=" + str(init_learning_rate) + "_bs=" + str(batch_size)

tag_tiling = "xp5h_BCE_LN__" + xp_param_string


#f_model_head_path_in = ""   # Pas de tete preentrainée, recréée à la volée
f_model_head_path_in = "tiling_base_model_head.pth"

filename_model_weights_out = "cp_" + tag_tiling + ".pth"

f_path_predictions_csv_out = "mads_split_test_predictions_" + tag_tiling + ".csv"

#deep_features_folder = "/home/vincent/Bureau/Bureau2/2_pipeline_quadrats/0_datastore/20_deep_features/mads_split/"
deep_features_folder = "/home/vincent/Bureau/Bureau2/4_pipeline_quadrats_officiel_danois/0_datastore/20_deep_features/DanishRoads/mads_split/"

#train_dataset_folder = deep_features_folder + "stdev_fixe_train"
#train_dataset_folder = deep_features_folder + "centre_reduit_train"
train_dataset_folder = deep_features_folder + "train"
#train_dataset_folder = deep_features_folder + "test_mini"
#train_dataset_folder = deep_features_folder + "test"

#val_dataset_folder = deep_features_folder + "test_mini"
#val_dataset_folder = deep_features_folder + "test"
val_dataset_folder = deep_features_folder + "val"
#val_dataset_folder = deep_features_folder + "stdev_fixe_val"
#val_dataset_folder = deep_features_folder + "centre_reduit_val"

test_dataset_folder = deep_features_folder + "test"
#test_dataset_folder = deep_features_folder + "stdev_fixe_test"
#test_dataset_folder = deep_features_folder + "centre_reduit_test"
#test_dataset_folder = deep_features_folder + "val"

tensorflow_log_dir = "runs"

bool_caching_features = True
#bool_caching_features = False

caching_device_name = "cpu"    # Caching en RAM
model_device_name = "cuda"     # Calculs sur GPU

n_views = 926
n_features = 768

#species_list = ["Heracleum", "Solidago", "Cytisus scoparius", "Rosa rugosa", "Lupinus polyphyllus", "Pastinaca sativa", "Reynoutria", "no_species"]
#species_list = ["Heracleum", "Solidago", "Cytisus scoparius", "Rosa rugosa", "Lupinus polyphyllus", "Pastinaca sativa", "Reynoutria"]
#species_list = ["Solidago", "Cytisus scoparius", "Rosa rugosa", "Lupinus polyphyllus", "Pastinaca sativa", "Reynoutria", "no_species"]
species_list = ["Solidago", "Cytisus scoparius", "Rosa rugosa", "Lupinus polyphyllus", "Pastinaca sativa", "Reynoutria"]

n_species = len(species_list)


#######################
## Chargement des librairies et définition des fonctions

import os
import re
import numpy as np
import torch

from lib_datasets import datasets_generic_ImageFolder, get_dataloader
from lib_learning import perform_learning
from lib_models import calc_nb_params_model, calc_nb_learnable_params_model
from lib_models_aggregation import create_model_weighted_sum_or_max_aggregation




#######################
## Code principal
print("###########################")
print("Debut du code principal")

print("####")
print("Paramètres utilisés:")

print("bool_use_BCE_loss :", bool_use_BCE_loss)
print("init_learning_rate :", init_learning_rate)
print("weight_decay :", weight_decay)
print("label_smoothing_factor :", label_smoothing_factor)
print("p_dropout :", p_dropout)
print("batch_size :", batch_size)
print("num_workers :", num_workers)
print("n_epochs :", n_epochs)
print("early_stop_patience :", early_stop_patience)
print("early_stop_min_delta :", early_stop_min_delta)
print("reduce_lr_patience :", reduce_lr_patience)
print("reduce_lr_factor :", reduce_lr_factor)
print("bool_add_softmax_layer :", bool_add_softmax_layer)
print("bool_max_logits :", bool_max_logits)
print("bool_linear_avant_max :", bool_linear_avant_max)
print("n_model_views :", n_model_views)
print("f_resume_in :", f_resume_in)
print("tag_tiling :", tag_tiling)
print("f_model_head_path_in : >" + f_model_head_path_in + "<")
print("filename_model_weights_out :", filename_model_weights_out)
print("f_path_predictions_csv_out :", f_path_predictions_csv_out)
print("train_dataset_folder :", train_dataset_folder)
print("val_dataset_folder :", val_dataset_folder)
print("test_dataset_folder :", test_dataset_folder)
print("bool_caching_features :", bool_caching_features)
print("caching_device_name :", caching_device_name)
print("model_device_name :", model_device_name)
print("n_views :", n_views)
print("n_features :", n_features)
print("species_list :", species_list)
print("n_species :", n_species)
print("####")



# Permet d'éviter que l'environnement CUDA freeze, et donne plus d'indications sur les erreurs CUDA.
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
# Déplacé dans lib_learning

#import time
#print("Sleep pendant 5 secondes..")
#time.sleep(5)

## Chargement du modèle
print("Création du modèle:")

if (f_model_head_path_in):
    init_classif_linear = torch.load(f_model_head_path_in, map_location = model_device_name)
else:
    init_classif_linear = None

[model, model_infos] = create_model_weighted_sum_or_max_aggregation(
            n_views    = n_views,
            n_model_views = n_model_views,
            n_features = n_features,
            n_species  = n_species,
            p_dropout  = p_dropout,

            bool_max_logits             = bool_max_logits,
            bool_argmax                 = False,
            bool_layer_norm             = bool_layer_norm,
            bool_max_add_softmax_layer  = bool_add_softmax_layer,
            bool_max_linear_avant_max   = bool_linear_avant_max,
            bool_final_sigmoid          = bool_use_BCE_loss,
            init_classif_linear         = init_classif_linear)


if (len(f_resume_in) > 0):
    model.load_state_dict(torch.load(f_resume_in), strict = True)
    print("Resume: Chargement des poids du modèle préentrainé", f_resume_in)

print(model)

model = model.to(torch.device(model_device_name))
#print(model.state_dict())
print("Nombre de paramètres:", calc_nb_params_model(model))
print("Nombre de paramètres appris:", calc_nb_learnable_params_model(model))


if (not bool_perform_learning):
    print("Apprentissage désactivé")
    exit()



## Chargement des données d'apprentissage
caching_device = torch.device(caching_device_name)

#transform_tensor = transforms.Compose([transforms.ToTensor()])
transform_tensor = torch.nn.Identity()

print("Création des datasets...")
if (n_epochs > 0):
    print("train_dataset")
    train_dataset = datasets_generic_ImageFolder(
                    dataset_folder                  = train_dataset_folder,
                    bool_return_data_images         = True,
                    bool_true_tensors_false_images  = True,
                    bool_caching_images             = bool_caching_features,
                    bool_caching_labels             = True,
                    bool_return_relative_img_names  = False,
                    bool_return_short_img_names     = False,
                    bool_one_hot_encoding           = bool_use_BCE_loss,
                    bool_one_hot_remove_last_col    = bool_use_BCE_loss,
                    transform                       = transform_tensor,
                    device                          = caching_device)
else:
    train_dataset = None

print("val_dataset")

val_dataset = datasets_generic_ImageFolder(
                dataset_folder                  = val_dataset_folder,
                bool_return_data_images         = True,
                bool_true_tensors_false_images  = True,
                bool_caching_images             = bool_caching_features,
                bool_caching_labels             = True,
                bool_return_relative_img_names  = False,
                bool_return_short_img_names     = False,
                bool_one_hot_encoding           = bool_use_BCE_loss,
                bool_one_hot_remove_last_col    = bool_use_BCE_loss,
                transform                       = transform_tensor,
                device                          = caching_device)

print("Création des dataloaders...")

if (n_epochs > 0):
    train_loader = get_dataloader(
                    dataset = train_dataset,
                    batch_size = batch_size,
                    shuffle = True,
                    num_workers = num_workers,
                    pin_memory = True)
else:
    train_loader = None


val_loader = get_dataloader(
                dataset = val_dataset,
                batch_size = batch_size,
                shuffle = False,
                num_workers = num_workers,
                pin_memory = True)


[train_loss_history,
    train_accuracy_history,
    val_loss_history,
    val_accuracy_history,
    duree_totale] = perform_learning(
                        model,
                        train_loader,
                        val_loader,
                        n_epochs,
                        init_learning_rate,
                        filename_model_weights_out,

                        early_stop_patience,
                        reduce_lr_patience,
                        early_stop_min_delta,
                        reduce_lr_factor,

                        label_smoothing_factor,
                        weight_decay,
                        bool_use_BCE_loss,
                        
                        bool_use_tensorboard,
                        tensorflow_log_dir,
                        tf_train_xp_name = xp_param_string + "__train",
                        tf_val_xp_name = xp_param_string + "__val")
