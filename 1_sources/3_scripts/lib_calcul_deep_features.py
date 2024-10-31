#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""This module provides code to compute the deep features (embeddings) of a set of images.

Author: Vincent Espitalier <vincent.espitalier@cirad.fr>
        Hervé Goëau <herve.goeau@cirad.fr>
"""

import os
import pathlib
import sys
import time
import math
import PIL.Image as Image

import torch
from torch.cuda.amp import autocast
from contextlib import contextmanager
from torch.utils.data import DataLoader, Dataset


import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import torchvision.transforms.functional as F

import kornia
from kornia.contrib import extract_tensor_patches
from kornia.enhance.normalize import Normalize, normalize

from lib_utils_files import write_to_json_file, save_torch_tensor

def affiche_tag_heure():
    from datetime import datetime
    import time
    now = datetime.now()
    print("tag heure: " + str(now.strftime("%Y-%m-%d_%H:%M:%S")) + " (" + str(time.time()) + ")")



def calc_dynamic_geometry(img_yx, model_input_size, overlap):
    
    img_y = img_yx[0]
    img_x = img_yx[1]

    dict_geo = {}
    dict_geo["version"]          = 1,
    dict_geo["resolution_y"]     = img_y
    dict_geo["resolution_x"]     = img_x

    dict_geo["model_input_size"] = model_input_size

    # For scales 1 to (n_scales - 1), we resize the image (and loose some information)
    # For the last scale, we use the true resolution of the image
    n_scales = math.ceil(min(img_y, img_x) / model_input_size)
    min_axe_is_y = (img_y < img_x)

    #n_scales = 3  # TODO : For debugging

    dict_geo["max_scale"] = n_scales
    dict_geo["scales_list"] = list(range(1, n_scales + 1))

    dict_geo["scales_n_views_yx_list"] = []
    dict_geo["n_views"] = 0
    dict_geo["strides_x"] = []
    dict_geo["strides_y"] = []
    for scale_index in range(1, n_scales + 1):

        if (min_axe_is_y):
            resized_res_y = scale_index * model_input_size
            resized_res_x = int(1. * resized_res_y * img_x / img_y)
        else:
            resized_res_x = scale_index * model_input_size
            resized_res_y = int(1. * resized_res_x * img_y / img_x)

        if ((resized_res_y > img_y) or (resized_res_x > img_x)):
            # True resolution of image
            resized_res_y = img_y
            resized_res_x = img_x


        n_views_y = math.ceil((resized_res_y - overlap * model_input_size) / ((1. - overlap) * model_input_size) )
        n_views_x = math.ceil((resized_res_x - overlap * model_input_size) / ((1. - overlap) * model_input_size) )

        if (n_views_y > 1):
            stride_y = int(1. * (resized_res_y - model_input_size) / (n_views_y - 1))
        else:
            stride_y = int((1. - overlap) * model_input_size)   # Default value

        if (n_views_x > 1):
            stride_x = int(1. * (resized_res_x - model_input_size) / (n_views_x - 1))
        else:
            stride_x = int((1. - overlap) * model_input_size)   # Default value

        if (False):
            print("scale index:", scale_index)
            print("resized_res_y, resized_res_x: ", resized_res_y, resized_res_x)
            print("n_views_y, n_views_x:", n_views_y, n_views_x)
            print("stride_y, stride_x:", stride_y, stride_x)
            real_overlap_y = int(100. * (model_input_size - stride_y) / model_input_size) / 100.
            real_overlap_x = int(100. * (model_input_size - stride_x) / model_input_size) / 100.
            print("real overlap:", real_overlap_y, real_overlap_x)
            last_y = 1 + (model_input_size - 1) + (n_views_y - 1) * stride_y
            last_x = 1 + (model_input_size - 1) + (n_views_x - 1) * stride_x
            print("resi_y, resi_x: ", resized_res_y, resized_res_x)
            print("last_y, last_x:", last_y, last_x)
            if (last_y > resized_res_y):
                print("Depasse sur y")
            if (last_x > resized_res_x):
                print("Depasse sur x")
            print("Nb pixels perdus:", resized_res_y - last_y, resized_res_x - last_x)
            print(" ")

        dict_geo["scales_n_views_yx_list"].append([n_views_y, n_views_x])
        dict_geo["strides_y"].append(stride_y)
        dict_geo["strides_x"].append(stride_x)
        dict_geo["n_views"] += n_views_y * n_views_x
    
    return dict_geo


def calc_tile_scale_index(dict_geometry, tile_index):
    if (tile_index < 0):
        return -1
    if (tile_index >= dict_geometry["n_views"]):
        return -1

    sum_n_views_yx = 0
    for scale_index in range(dict_geometry["max_scale"]):
        n_views_yx = dict_geometry["scales_n_views_yx_list"][scale_index]
        next_sum_n_views_yx = sum_n_views_yx + n_views_yx[0] * n_views_yx[1]
        if (next_sum_n_views_yx > tile_index):
            return (scale_index, tile_index - sum_n_views_yx)
        sum_n_views_yx = next_sum_n_views_yx
    return -1

# Output format: [x_min, y_min, x_max, y_max]
def calc_tile_coordinates(dict_geometry, tile_index):

    img_y            = dict_geometry["resolution_y"]
    img_x            = dict_geometry["resolution_x"]
    model_input_size = dict_geometry["model_input_size"]

    (scale_index, tile_index_in_scale)  = calc_tile_scale_index(dict_geometry, tile_index)
    if (scale_index < 0):
        print("Erreur dans calc_tile_scale_index:", tile_index, dict_geometry)
        return
    scale_index += 1   # Meme signification que la fonction qui construit la géométrie des vues

    stride_x                            = dict_geometry["strides_x"][scale_index - 1]
    stride_y                            = dict_geometry["strides_y"][scale_index - 1]
    [n_views_y, n_views_x]              = dict_geometry["scales_n_views_yx_list"][scale_index - 1]

    min_axe_is_y = (img_y < img_x)
    if (min_axe_is_y):
        resized_res_y = scale_index * model_input_size
        resized_res_x = int(1. * resized_res_y * img_x / img_y)
    else:
        resized_res_x = scale_index * model_input_size
        resized_res_y = int(1. * resized_res_x * img_y / img_x)

    if ((resized_res_y > img_y) or (resized_res_x > img_x)):
        # True resolution of image
        resized_res_y = img_y
        resized_res_x = img_x

    delta_x = int(img_x * model_input_size / resized_res_x)
    delta_y = int(img_y * model_input_size / resized_res_y)

    # On travaille dans le système de coordonnées d'origine de l'image
    # Pas celui de la nouvelle échelle avec l'image réduite
    stride_x = stride_x * img_x / resized_res_x
    stride_y = stride_y * img_y / resized_res_y

    #x_index = int(tile_index_in_scale / n_views_y)
    #y_index = tile_index_in_scale % n_views_y
    # Peut etre l'inverse
    x_index = tile_index_in_scale % n_views_x
    y_index = int(tile_index_in_scale / n_views_x)

    x_min = int(x_index * stride_x)
    y_min = int(y_index * stride_y)
    x_max = x_min + delta_x - 1
    y_max = y_min + delta_y - 1

    #x_min = 200; y_min = 400; x_max = 300; y_max = 700   # Pour tests
    return [x_min, y_min, x_max, y_max]


def make_dense_predictions_and_aggregate(
    model_name,
    model,
    model_input_size,
    model_embedding_dim,
    data_resized_image,
    data_preproc_img_mean,
    data_preproc_img_std,
    strides_yx,
    scale,
    scales_n_views_yx,
    param_batch_size_views,
    param_display_view,
    device,
    index_features,
    image_features_cls_token,
    image_features_moy_local_tokens):

    assert(not torch.isnan(data_resized_image).any())

    # Découpage de l'image en "views" avec overlapping (i.e. patches)
    views = extract_tensor_patches(
        input = data_resized_image,
        window_size = model_input_size,
        stride = strides_yx,
        #stride = 10,
        padding = 0) 

    assert(not torch.isnan(views).any())

    resized_height = data_resized_image.shape[2]
    resized_width = data_resized_image.shape[3]

    views_nb_y = scales_n_views_yx[0]
    views_nb_x = scales_n_views_yx[1]
    #print("Verification: views.shape[1] = ", views.shape[1], ", views_nb_y * views_nb_x = ", (views_nb_y * views_nb_x))
    if (views.shape[1] != (views_nb_y * views_nb_x)):
        print("Erreur dans le nombre de vues: Inconsistance entre l'output de Kornia.extract_tensor_patches() et les nb_y nb_x en entrée.")
        print("scale:", scale)
        print("data_resized_image.shape : ", data_resized_image.shape)
        print("views.shape : ", views.shape)
        print("views_nb_y , views_nb_x : ", views_nb_y , views_nb_x)
        print("resized_height , resized_width : ", resized_height , resized_width)
        print("strides_yx : ", strides_yx)
        exit(-1)

    positions = []
    for y in range(views_nb_y):
        pos_y = y * strides_yx[0]
        for x in range(views_nb_x):
            pos_x = x * strides_yx[1]
            positions.append([pos_y, pos_x])

            #print(pos_y, pos_x)
    #print('make_dense_predictions_and_aggregate_futur: big image subdivided into', sub_imagess.shape, 'image patches, from a stride_patch:', stride_patch, ', ', subimage_nb_y, 'subimages along Y', len(positions), 'positions')
           
    nb_positions_in_one_image = len(positions)
    batch_nb = math.ceil(nb_positions_in_one_image / param_batch_size_views)

    
    #remove first dimension related to the batch, because forward need a continous batch
    views = views.reshape(views.shape[0] * views.shape[1], views.shape[2], views.shape[3], views.shape[4])
    #print('views:', views.shape)
    
    #print('will perform', batch_nb, 'batch predictions of', param_batch_size_views, ' views;  views shape: ', views.shape)    
    
    model.eval()
    for i in range(batch_nb):
        start = i * param_batch_size_views
        end = start + param_batch_size_views
        #print('start:', start, ', end:', end)

        current_views = views[start:end, :, :, :]
        #print("current_views.shape:", current_views.shape)
        current_positions = positions[start:end]
        
        if param_display_view:
            for j in range(len(views)):# just big image from batchid 0
                if j== 0:
                    vi = views[j]
                    #print('current_batch:', i, ', current view', j, vi.shape, 'at position', current_positions[j])
                    #display_image(si, display=True)#, display_size=6)

        assert(not torch.isnan(current_views).any())
        current_views = Normalize(data_preproc_img_mean, data_preproc_img_std)(current_views)   
        assert(not torch.isnan(current_views).any())
        #print("current_views.shape:", current_views.shape)
        features = model.forward_features(current_views)
        assert(not torch.isnan(features).any())
        #if (model_name[0:4] == "beit"):
        #    if model.fc_norm is not None:
        #        features = features[:, 1:].mean(dim=1)
        #        features = model.fc_norm(features)
        #    else:
        #        features = features[:, 0]
        
        features_cls_token = features[:, 0]
        features_moy_local_tokens = features[:, 1:].mean(dim=1)
        if model.fc_norm is not None:
            features_moy_local_tokens = model.fc_norm(features_moy_local_tokens)
        assert(not torch.isnan(features_cls_token).any())
        assert(not torch.isnan(features_moy_local_tokens).any())
        
        #print("features.shape:", features.shape)
        #print("image_features.shape:", image_features.shape)
        d1 = features_cls_token.shape[0]
        #print("debug:", d1, index_features, start)
        #image_features[(index_features + start):(index_features + start + d1), :] = features
        image_features_cls_token[(index_features + start):(index_features + start + d1), :]        = features_cls_token
        image_features_moy_local_tokens[(index_features + start):(index_features + start + d1), :] = features_moy_local_tokens
    



def resize_image(input_image, target_image_size, device):
    input_image = input_image.to(device, non_blocking=True)
    _, _, input_height, input_width = input_image.shape
    # Force le resize, pour adapter la résolution de l'image.
    #if target_image_size != input_height or target_image_size != input_width:
    output_image = F.resize(input_image, size = target_image_size)
    return output_image



# Data input image must be a batch of one image
def make_hierarchical_features_predictions_rectangular_picture(
    model_name,
    model,
    model_input_size,
    model_embedding_dim,
    data_input_image,
    data_preproc_img_mean,
    data_preproc_img_std,
    param_scales_list,
    param_scales_strides_list,
    param_scales_n_views_yx_list,
    param_batch_size_views,
    param_display_view,
    device):

    n_views_per_image = sum([param_scales_n_views_yx_list[i][0] * param_scales_n_views_yx_list[i][1] for i in range(len(param_scales_list))])
    #print("n_views_per_image:", n_views_per_image)
    image_features_cls_token        = torch.zeros(n_views_per_image, model_embedding_dim, device=torch.device('cuda'))
    image_features_moy_local_tokens = torch.zeros(n_views_per_image, model_embedding_dim, device=torch.device('cuda'))

    index = 0    
    index_features = 0
    for index in range(len(param_scales_list)):
        scale             = param_scales_list[index]
        strides_yx        = param_scales_strides_list[index]
        scales_n_views_yx = param_scales_n_views_yx_list[index]

        #if (scale != len(param_scales_list)):
        if (True):
            rescaled_image_size = model_input_size * scale
            data_resized_image = resize_image(data_input_image, rescaled_image_size, device)
        #else:
        #    rescaled_image_size = data_input_image.shape[2:4]
        #    print(rescaled_image_size)
        #    data_resized_image = resize_image(data_input_image, rescaled_image_size, device)

        
        n_views_curr_scale = scales_n_views_yx[0] * scales_n_views_yx[1]

        print('make_hierarchical_features_predictions_rectangular_picture: ',\
                    '  scale:',              scale, \
                    ', rescaled_image_size', rescaled_image_size, \
                    ', strides:',            strides_yx, \
                    ', n_views_curr_scale:', n_views_curr_scale)
        if (False):
            print(torch.cuda.memory_summary())
            
        sys.stdout.flush()

        make_dense_predictions_and_aggregate(
            model_name,
            model,
            model_input_size,
            model_embedding_dim,
            data_resized_image,
            data_preproc_img_mean,
            data_preproc_img_std,
            strides_yx,
            scale,
            scales_n_views_yx,
            param_batch_size_views,
            param_display_view,
            device,
            index_features,
            image_features_cls_token,
            image_features_moy_local_tokens)

        index_features = index_features + n_views_curr_scale
        #print("index_features ", index_features, "a été incrémenté de ", n_views_curr_scale)

    return [ image_features_cls_token, image_features_moy_local_tokens ]


def calcule_deep_features(
        model,
        model_infos,
        dataloader,
        views_geometry,
        batch_size_views,
        output_deep_features_folder,
        use_autocast,
        use_torch_compile,
        device):

    print("Appel à calcule_deep_features()")
    # Afficher les versions des principales librairies utilisées
    print("Version de python:", sys.version_info)
    print("Version de pytorch:", torch.__version__)
    print("Version de torchvision:", torchvision.__version__)
    print("Version de kornia:", kornia.__version__)

    model_name          = model_infos["model_name"]
    model_input_size    = model_infos["model_input_size"]
    model_embedding_dim = model_infos["embedding_dim"]
    img_mean            = model_infos["img_mean"]
    img_std             = model_infos["img_std"]

    calcul_inferences = True
    #calcul_inferences = False  # Pour debuggage
    
    if (use_autocast):
        print("autocast activé")
    else:
        print("autocast désactivé")
        none_context = contextmanager(lambda: iter([None]))()
    
    # 1) model: Compile si demandé
    # https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html
    # Requires 'triton'  -> $ conda install pytorch::torchtriton
    if (use_torch_compile):
        try:
            print("torch compile activé")
            
            model_opt = torch.compile(model)
        except Exception as error:
            print("Warning: Le modèle n'a pas pu être compilé.")
            print(">>>>>>>>")
            print("Erreur:", type(error).__name__, " – ", error)
            print("<<<<<<<<")
            print("Poursuite des calculs sans torch compilation.")
            model_opt = model
    else:
        print("torch compile désactivé")
        model_opt = model

    model_opt.eval()
    model_opt = model_opt.to(device)    

    # 2) Charger les données
    n_images = len(dataloader.dataset)
    print("n_images:", n_images)
    n_batch = len(dataloader)

    #output_deep_features_cls_token_folder        = os.path.join(output_deep_features_folder, "cls_token")
    #output_deep_features_moy_local_tokens_folder = os.path.join(output_deep_features_folder, "moy_local_tokens")

    # We only save the moy_local_token, in the main directory
    output_deep_features_cls_token_folder        = ""
    output_deep_features_moy_local_tokens_folder = output_deep_features_folder
    
    
    if (False):
        print(output_deep_features_folder)
        print(output_deep_features_cls_token_folder)
        print(output_deep_features_moy_local_tokens_folder)
        exit()

    print(" ")
    print("Lancement des inférences")

    # Geométries des vues fournies
    if (views_geometry != "dynamic"):
        n_scales_list          = views_geometry["max_scale"]
        scales_n_views_yx_list = views_geometry["scales_n_views_yx_list"]
        scales_list            = views_geometry["scales_list"]
        scales_strides_list    = [ [views_geometry["strides_y"][i - 1], views_geometry["strides_x"][i - 1] ] for i in scales_list]
        n_views_per_image = sum([scales_n_views_yx_list[i][0] * scales_n_views_yx_list[i][1] for i in range(len(scales_list))])

        taille_deep_features_Mo = (4 * n_images * n_views_per_image * model_embedding_dim) / 1024 / 1024
        print("Taille totale deep features sur disque dur (Mo) :", taille_deep_features_Mo)

    time_begin = time.time()
    with torch.no_grad():
        with (autocast() if use_autocast else none_context):
            index_batch = 0
            #time_begin = time.time()
            for batch_image, batch_img_relative_loc in dataloader:
                print(batch_img_relative_loc[0])
                #batch_image = Normalize(img_mean, img_std)(batch_image.to(device))
                batch_image.to(device)
                f_path_out_predictions_cls_token        = \
                    os.path.join(output_deep_features_cls_token_folder, batch_img_relative_loc[0] + ".pth")
                f_path_out_predictions_moy_local_tokens = \
                    os.path.join(output_deep_features_moy_local_tokens_folder, batch_img_relative_loc[0] + ".pth")

                time_prec = time.time()
                file_exists = os.path.isfile(f_path_out_predictions_moy_local_tokens)
                #file_exists = False   # Pour debuggage
                if (file_exists):
                    print("File exists: ", f_path_out_predictions_moy_local_tokens)
                    print("Going to analyse next image ... ")
                    time_now = time.time()
                else:
                    # Geométries des vues pas fournies => Calcul dynamique de la geometrie des vues
                    if (views_geometry == "dynamic"):
                        img_yx = list(batch_image.shape[2:4])
                        print(img_yx, batch_img_relative_loc[0])
                        overlap = 0.50
                        #extra_scale_magnif = 1.25
                        extra_scale_magnif = 0.0
                        views_geometry_dynamic = calc_dynamic_geometry(img_yx, model_input_size = model_input_size, overlap = 0.50)
                        scales_n_views_yx_list = views_geometry_dynamic["scales_n_views_yx_list"]
                        scales_list            = views_geometry_dynamic["scales_list"]
                        scales_strides_list    = [[views_geometry_dynamic["strides_y"][i - 1], \
                                                   views_geometry_dynamic["strides_x"][i - 1]] for i in scales_list]
                        print("Dynamic tiles geometry computation:")
                        print(scales_n_views_yx_list)
                        print(scales_list)
                        print(scales_strides_list)
                        #print("debug > exit")
                        #exit()

                    if (calcul_inferences):
                        [ image_features_cls_token, image_features_moy_local_tokens ] =  \
                            make_hierarchical_features_predictions_rectangular_picture(
                                model_name = model_name,
                                model = model_opt,
                                model_input_size = model_input_size,
                                model_embedding_dim = model_embedding_dim,
                                data_input_image = batch_image,
                                data_preproc_img_mean = img_mean,
                                data_preproc_img_std = img_std,
                                param_scales_list = scales_list,
                                param_scales_strides_list = scales_strides_list,
                                param_scales_n_views_yx_list = scales_n_views_yx_list,
                                param_batch_size_views = batch_size_views,
                                param_display_view = False,
                                device = device)
                        

                        # 4) Sauvegarder les tenseurs : cls token + moy local tokens
                        if (len(output_deep_features_cls_token_folder) > 1):
                            folder_cls_token = os.path.dirname(f_path_out_predictions_cls_token)
                            print("debug 267:", f_path_out_predictions_cls_token)
                            print("debug 267b:", folder_cls_token)
                            if (not os.path.exists(folder_cls_token)):
                                # Create sub folder if does not exist
                                print("Creation of folder:", folder_cls_token)
                                pathlib.Path(folder_cls_token).mkdir(parents=True, exist_ok=False)
                            save_torch_tensor(image_features_cls_token,        f_path_out_predictions_cls_token)

                        if (len(output_deep_features_moy_local_tokens_folder) > 1):
                            folder_moy_local_tokens = os.path.dirname(f_path_out_predictions_moy_local_tokens)
                            print("debug 268:", f_path_out_predictions_moy_local_tokens)
                            print("debug 268b:", folder_moy_local_tokens)
                            if (not os.path.exists(folder_moy_local_tokens)):
                                # Create sub folder if does not exist
                                print("Creation of folder:", folder_moy_local_tokens)
                                pathlib.Path(folder_moy_local_tokens).mkdir(parents=True, exist_ok=False)
                            save_torch_tensor(image_features_moy_local_tokens, f_path_out_predictions_moy_local_tokens)
                            
                    time_now = time.time()
                    
                    # 5) Enregistre aussi la géométrie de l'image si dynamique
                    if (views_geometry == "dynamic"):
                        f_path_out_geometry_image = f_path_out_predictions_cls_token + "_geometry.json"
                        write_to_json_file(f_path_out_geometry_image, views_geometry_dynamic)

                #print(features)
                #print(index_batch, image_features.size())
                index_batch = index_batch + 1
                duration = (time_now - time_prec)
                affiche_tag_heure()
                print("Durée inférence par image en secondes:", duration)
                duration_total = (time_now - time_begin)
                duration_par_batch = duration_total / index_batch
                duration_remain_min = int(((n_batch - index_batch) * duration_par_batch) / 60.)
                duration_remain_hrs = int(duration_remain_min / 60. * 10.) / 10.
                print("Durée par batch en secondes:", duration_par_batch)
                print("Index batch:", index_batch, " sur ", n_batch)
                print("Duree restante: ", duration_remain_hrs, "hours = ", duration_remain_min, "mins")
                sys.stdout.flush()

