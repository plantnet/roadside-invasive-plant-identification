#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""This module provides the code for linking the libs to the generic pipeline. The wrapping structures are called "modules"

Author: Vincent Espitalier <vincent.espitalier@cirad.fr>
        Hervé Goëau <herve.goeau@cirad.fr>
"""

# No include to external library
# The aim of this file is to "link" and call the pipeline modules

from lib_utils_files import read_text_file, read_list_text_files, write_text_file, read_text_integer_file, read_json_file, write_to_json_file, pretty_json, load_torch_tensor, load_torch_tensor_list, save_torch_tensor, save_model_state_dict, load_model_state_dict, save_model_head, charge_fichier_csv_vers_dataframe, charge_liste_fichiers_csv_vers_dataframes, enregistre_dataframe_vers_fichier_csv, enregistre_dataframe_vers_fichier_csv_excel, get_list_of_files_in_directory, create_folder_if_does_not_exists, randomSleep

from lib_datasets import create_liste_augmentations_from_model_infos, \
        get_augmentations, get_dataloader, \
        datasets_generic_ImageFolder, simple_dataset, datasets_generic_ArbitraryFolder, get_annotations_data

from lib_models import load_model_timm, load_aggregation_model, model_centrer_reduire_sans_biais

from lib_calcul_deep_features import calcule_deep_features

from lib_learning import get_device, perform_learning, calc_predictions_and_output_tensor_and_images_names

from lib_calcul_stats_binaires import wrappe_predictions_in_dataframe_with_common_names_and_tensors_list, calculate_binary_stats


##########################################
# Modules liés aux fichiers (lib_utils_files)

# Fichiers: Lecture / écriture de fichiers textes
def module_read_text_file(params, data):
    return read_text_file(f_path = params["f_path"])
    
def module_read_list_text_files(params, data):
    return read_list_text_files(
                folder      = params["folder"],
                f_path_list = params["f_path_list"])
                
def module_write_text_file(params, data):
    write_text_file(
            my_string   = params["my_string"],
            f_path      = params["f_path"])

def module_read_text_integer_file(params, data):
    return read_text_integer_file(f_path = params["f_path"])


# Fichiers: Lecture / écriture de fichiers json
def module_read_json_file(params, data):
    return read_json_file(
    f_path = params["f_path"])
 
def module_write_to_json_file(params, data):
    return write_to_json_file(
    f_path = params["f_path"],
    dictionary = params["dict"])

def module_pretty_json(params, data):
    return pretty_json(
    dictionary = params["dictionary"])


# Fichiers: Lecture / écriture de tenseurs
def module_load_torch_tensor(params, data):
    return load_torch_tensor(f_path = params["f_path"],
                             device = data["device"])

def module_load_torch_tensor_list(params, data):
    return load_torch_tensor_list(
        tensors_folder  = params["tensors_folder"],
        dict_f_paths    = params["dict_f_paths"],
                 device = data["device"])

def module_save_torch_tensor(params, data):
    save_torch_tensor(
        tensor = data["tensor"],
        f_path = params["f_path"])


# Fichiers: Lecture / écriture de state_dict de modeles
def module_save_model_state_dict(params, data):
    return save_model_state_dict(
        model = data["model"],
        filename_model_weights_out = params["f_path"])

def module_load_model_state_dict(params, data):
    return load_model_state_dict(
        model = data["model"],
        filename_model_weights_in = params["f_path"])

def module_save_model_head(params, data):
    return save_model_head(
        model                   = data["model"],
        f_model_head_path       = params["f_path"])


# Fichiers: Lecture / écriture de dataframes
def module_charge_fichier_csv_vers_dataframe(params, data):
    return charge_fichier_csv_vers_dataframe(f_path = params["f_path"])

def module_charge_liste_fichiers_csv_vers_dataframes(params, data):
    return charge_liste_fichiers_csv_vers_dataframes(
                folder_statistiques = params["folder_statistiques"],
                dict_f_paths        = params["dict_f_paths"])

def module_enregistre_dataframe_vers_fichier_csv(params, data):
    return enregistre_dataframe_vers_fichier_csv(
        df = data["df"],
        f_path = params["f_path"])

def module_enregistre_dataframe_vers_fichier_csv_excel(params, data):
    if ("export_index" in params.keys()):
        return enregistre_dataframe_vers_fichier_csv_excel(
            df = data["df"],
            f_path = params["f_path"],
            export_index = params["export_index"])
    else:
        # Rétro-compatibilité, au cas ou le paramètre est manquant
        return enregistre_dataframe_vers_fichier_csv_excel(
            df = data["df"],
            f_path = params["f_path"],
            export_index = True)


# Fichiers: Divers
def module_get_list_of_files_in_directory(params, data):
    return get_list_of_files_in_directory(
        directory      = params["directory"],
        file_extension = params["file_extension"])



##########################################
# Modules liés aux données (lib_datasets)
def module_create_liste_augmentations_from_model_infos(params, data):
    return create_liste_augmentations_from_model_infos(
        model_infos          = params["model_infos"],
        create_augment_specs = params["create_augment_specs"])

def module_get_augmentations(params, data):
    liste_augmentations = params["liste_augmentations"]
    return get_augmentations(liste_augmentations)


def module_get_dataloader(params, data):
    dataset     = data["dataset"]
    batch_size  = params["batch_size"]
    shuffle     = params["shuffle"]
    num_workers = params["num_workers"]
    pin_memory  = params["pin_memory"]
    return get_dataloader(dataset, batch_size, shuffle, num_workers, pin_memory)

def module_datasets_generic_ImageFolder(params, data):
    return datasets_generic_ImageFolder(
        dataset_folder                  = params["dataset_folder"],
        transform                       = params["augmentations"],
        device                          = data["device"],
        bool_true_tensors_false_images  = params["bool_true_tensors_false_images"],
        bool_caching_images             = params["bool_caching_images"],
        bool_caching_labels             = params["bool_caching_labels"],
        bool_return_relative_img_names  = params["bool_return_relative_img_names"],
        bool_return_short_img_names     = params["bool_return_short_img_names"],
        bool_one_hot_encoding           = params["bool_one_hot_encoding"],
        bool_one_hot_remove_last_col    = params["bool_one_hot_remove_last_col"])

def module_datasets_generic_ArbitraryFolder(params, data):
    return datasets_generic_ArbitraryFolder(
        dataset_folder                  = params["dataset_folder"],
        transform                       = params["augmentations"],
        device                          = data["device"],
        bool_true_tensors_false_images  = params["bool_true_tensors_false_images"],
        bool_caching_images             = params["bool_caching_images"],
        bool_return_relative_img_names  = params["bool_return_relative_img_names"],
        bool_return_short_img_names     = params["bool_return_short_img_names"])

def module_simple_dataset(params, data):
    return simple_dataset(
        main_dir                        = params["dataset_folder"],
        transform                       = params["augmentations"],
        images_extension                = params["images_extension"])

def module_get_annotations_data(params, data):
    return get_annotations_data(
        dataset_folder                  = params["dataset_folder"],
        dict_classes_folder_to_name     = params["dict_classes_folder_to_name"])


##########################################
# Modules de lib_modules_implementation
def module_get_device(params, data):
    return get_device(device_name = params["device_name"])

def module_wrappe_predictions_in_dataframe_with_common_names_and_tensors_list(params, data):
    return wrappe_predictions_in_dataframe_with_common_names_and_tensors_list(
                aggregated_probas = data["aggregated_probas"],
                liste_noms_communs = params["liste_noms_communs"],
                tensors_list = params["tensors_list"])


##########################################
# Modules liés aux modèles (lib_models_timm, lib_models)

def module_load_model_timm(params, data):
    return load_model_timm(
        model_name      = params["model_name"],
        models_infos    = params["models_infos"],
        model_input_size_xy_vamis = params["model_input_size_xy_vamis"],
        d_models_folder = params["d_models_folder"],
        device          = data["device"])

def module_load_aggregation_model(params, data):
    return load_aggregation_model(
        aggregation_type            = params["aggregation_type"],
        n_views                     = params["n_views"],
        n_model_views               = params["n_model_views"],
        n_features                  = params["n_features"],
        n_species                   = params["n_species"],
        p_dropout                   = params["p_dropout"],
        bool_argmax                 = params["bool_argmax"],
        bool_layer_norm             = params["bool_layer_norm"],
        bool_max_add_softmax_layer  = params["bool_max_add_softmax_layer"],
        bool_max_linear_avant_max   = params["bool_max_linear_avant_max"],
        bool_final_sigmoid          = params["bool_final_sigmoid"],
        bool_attention_feed_forward = params["bool_attention_feed_forward"],
        init_classif_linear         = data["init_classif_linear"])

def module_model_centrer_reduire_sans_biais(params, data):
    return model_centrer_reduire_sans_biais(
        f_model_in                       = params["f_model_in"],
        f_model_out                      = params["f_model_out"])


##########################################
# Modules liés aux inférences, learnings et calculs de stats
# lib_calcul_deep_features, lib_learning, lib_inference, lib_calcul_stats_binaires
def module_calcule_deep_features(params, data):
    return calcule_deep_features(
        model                       = data["model"],
        model_infos                 = params["model_infos"],
        dataloader                  = data["dataloader"],
        views_geometry              = params["views_geometry"],
        batch_size_views            = params["batch_size_views"],
        output_deep_features_folder = params["output_deep_features_folder"],
        use_autocast                = params["use_autocast"],
        use_torch_compile           = params["use_torch_compile"],
        device                      = data["device"])


def module_perform_learning(params, data):
    return perform_learning(
        model                       = data["model"],
        train_loader                = data["train_loader"],
        val_loader                  = data["val_loader"],
        n_epochs                    = params["n_epochs"],

        init_learning_rate          = params["init_learning_rate"],
        filename_model_weights_out  = params["filename_model_weights_out"],

        early_stop_patience         = params["early_stop_patience"],
        reduce_lr_patience          = params["reduce_lr_patience"],
        early_stop_min_delta        = params["early_stop_min_delta"],
        reduce_lr_factor            = params["reduce_lr_factor"],

        label_smoothing_factor      = params["label_smoothing_factor"],
        weight_decay                = params["weight_decay"],
        bool_use_BCE_loss           = params["bool_use_BCE_loss"],
        
        bool_use_tensorboard        = params["bool_use_tensorboard"],
        tensorflow_log_dir          = params["tensorflow_log_dir"],
        tf_train_xp_name            = params["tf_train_xp_name"],
        tf_val_xp_name              = params["tf_val_xp_name"])


def module_calc_predictions_and_output_tensor_and_images_names(params, data):
    return calc_predictions_and_output_tensor_and_images_names(
        model             = data["model"],
        data_loader       = data["data_loader"],
        use_autocast      = params["use_autocast"],
        use_torch_compile = params["use_torch_compile"],
        device            = data["device"])


def module_calculate_binary_stats(params, data):
    return calculate_binary_stats(
        input_df_annotations         =   data["input_df_annotations"],
        input_df_predictions         =   data["input_df_predictions"],
        output_f_path_images_prefixe = params["images_prefixe"],
        params_list_names_classes    = params["list_names_classes"],
        
        f_path_seuils_detection      = params["f_path_seuils_detection"],
        input_seuils_detection       = params["input_seuils_detection"],
        output_seuils_detection      = params["output_seuils_detection"],
        
        params_list_statistiques_for_export = params["list_statistiques_for_export"],
        interp_densite               = params["interp_densite"])



#####################

# Generic function which calls all the other functions above respectively with module_name
# All individual modules are called by this one
def module_pipeline_generic_function(module_name, local_params, local_data):
    if (not (module_name in globals().keys())):
        print("Error in lib_module_interface: module is absent:", module_name)
        exit(-1)

    #module_func = getattr(modules_post_treatment, module_name)
    module_func = globals()[module_name]
    module_output = module_func(local_params, local_data)
    #print("type module_output:", type(module_output))
    return module_output
