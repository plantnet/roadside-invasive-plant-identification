#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""This module provides code to manage files: loads and save for various kinds of files (text files, json, dataframe, tensors..)

Author: Vincent Espitalier <vincent.espitalier@cirad.fr>
        Hervé Goëau <herve.goeau@cirad.fr>
"""

import os
import json
import torch
import pandas as pd
import time
import numpy


#######################################
# Lecture / écriture de fichiers textes

# Retourne la liste des lignes d'un fichier texte
# sous la forme d'une liste de chaines
def read_text_file(f_path):
    str_list = []
    with open(f_path, 'r') as f:
        for line in f:
            #str_list.append(str(line))
            str_list.append(line.replace("\n", ""))
    return str_list

def read_list_text_files(folder, f_path_list):
    output = {}
    for f_path in f_path_list:
        f_full_path_csv = os.path.join(folder, f_path)
        #print("folder", folder, "f_path", f_path)
        output[f_path] = read_text_file(f_full_path_csv)
    return output

# (La chaîne contient déjà des retours à la ligne éventuels)
def write_text_file(my_string, f_path):
    with open(f_path, 'w') as file:
        file.write(my_string)
    print("Ecriture du fichier:", f_path)

# Retourne la liste d'entiers contenus dans un fichier texte
# Un nombre par ligne
def read_text_integer_file(f_path):
    int_list = []
    str_list = read_text_file(f_path)
    for str_curr in str_list:
        int_list.append(int(str_curr))
    return int_list
    

#######################################
# Lecture / écriture de fichiers json
    
def read_json_file(f_path):
    print("Ouverture du fichier :", f_path)
    with open(f_path, 'r') as j:
        return json.loads(j.read())

def write_to_json_file(f_path, dictionary):
    serialized_json = json.dumps(dictionary, indent = 2)
    write_text_file(serialized_json, f_path)

def pretty_json(dictionary):
    return json.dumps(dictionary, indent = 2)


#######################################
# Lecture / écriture de tenseurs

def load_torch_tensor(f_path, device):
    if (os.path.exists(f_path)):
        #print("load_torch_tensor:", f_path)
        return torch.load(f = f_path, map_location = device)
    else:
        print("Impossible de lire le fichier, car manquant:", f_path)
        exit(-1)
    
def load_torch_tensor_list(tensors_folder, dict_or_list_f_paths, device):

    if isinstance(dict_or_list_f_paths, dict):
        dict_tensors = {}
        for key, value in dict_or_list_f_paths.items():
            f_full_path_tensor = os.path.join(tensors_folder, value)
            dict_tensors[key] = load_torch_tensor(f_full_path_tensor, device)
        return dict_tensors

    if isinstance(dict_or_list_f_paths, list):
        list_tensors = []
        for value in dict_or_list_f_paths:
            f_full_path_tensor = os.path.join(tensors_folder, value)
            list_tensors.append(load_torch_tensor(f_full_path_tensor, device))
        return list_tensors

    print("load_torch_tensor_list : Error : Unrecognized type for dict_or_list_f_paths")
    return None


def save_torch_tensor(tensor, f_path):
    torch.save(tensor, f_path)
    print("Ecriture du fichier:", f_path)


#######################################
# Lecture / écriture de state_dict de modeles

def save_model_state_dict(model, filename_model_weights_out):
    torch.save(model.state_dict(), filename_model_weights_out)
    print("Sauve le state dict:", filename_model_weights_out)

def load_model_state_dict(model, filename_model_weights_in):
    if (len(filename_model_weights_in) > 0):
        print("Charge le state dict:", filename_model_weights_in)
        model.load_state_dict(torch.load(filename_model_weights_in), strict = True)
    else:
        print("(warning) load_model_state_dict : Pas de chargement de state_dict (nom de fichier vide).")


def save_model_head(model, f_model_head_path):
    torch.save(model.head, f_model_head_path)
    print("Fichier enregistré:", f_model_head_path)


#######################################
# Lecture / écriture de dataframes

# Generic functions which loads a .csv to a pandas dataFrame,
# and the reverse one :   saves a pandas dataFrame to a .csv 
def charge_fichier_csv_vers_dataframe(f_path, sep = ';'):
    print("Lecture du fichier:", f_path)
    return pd.read_csv(f_path, delimiter = sep)

def charge_liste_fichiers_csv_vers_dataframes(folder_statistiques, dict_f_paths, sep = ';'):
    dict_df = {}
    for key, value in dict_f_paths.items():
        f_full_path_csv = os.path.join(folder_statistiques, value)
        dict_df[key] = charge_fichier_csv_vers_dataframe(f_full_path_csv)
    print("charge_liste_fichiers_csv_vers_dataframes:")
    print(dict_df)
    return dict_df

def enregistre_dataframe_vers_fichier_csv(df, f_path, export_index = True, sep = ';'):
    if (df is None):
        print("Warning - enregistre_dataframe_vers_fichier_csv : dataframe is None. Not saving the file")
    else:
        df.to_csv(f_path, sep = sep, index = export_index)
        print("Ecriture du fichier:", f_path)

# We record the dataFrame by replacing the "." to "," (In LibreOffice, numbers use comma)
def enregistre_dataframe_vers_fichier_csv_excel(df, f_path, export_index = True, sep = ';'):
    df_excel = df.applymap(lambda x: str(x).replace(".", ",") if isinstance(x, float) else x)
    enregistre_dataframe_vers_fichier_csv(df_excel, f_path, export_index, sep)


#######################################
# Divers

# Source: https://stackoverflow.com/questions/9816816/get-absolute-paths-of-all-files-in-a-directory
def get_list_of_files_in_directory(directory, file_extension = ".pth"):
    file_path_list = []
    print("Listing du dossier:", directory)
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            if (f.endswith(file_extension)):
                f_abspath = os.path.abspath(os.path.join(dirpath, f))
                file_path_list.append(f_abspath)
    file_path_list.sort()
    return file_path_list

def create_folder_if_does_not_exists(directory):
    if (not(os.path.exists(directory))):
        os.makedirs(directory)

def randomSleep(min_time, max_time):
    duration = numpy.random.uniform(low = min_time, high = max_time)
    print("Waiting time", duration)
    time.sleep(duration)
