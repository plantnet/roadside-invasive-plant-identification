#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""This module provides the logic for the generic pipeline, which calls all the successive modules listed in json pipelines

Author: Vincent Espitalier <vincent.espitalier@cirad.fr>
        Hervé Goëau <herve.goeau@cirad.fr>
"""

f_pipeline_paths = "pipeline_paths.json"
verbose = True

import os, sys, json
import re
import time
from datetime import datetime
from pipeline_interface import module_pipeline_generic_function

########################


def module_include(params, data):
    f_relative_include_pipeline_json = params["pipeline"]

    # 1) Ouvrir le fichier json et charger le pipeline
    f_pipeline_paths        = "pipeline_paths.json"
    pipeline_paths          = check_exists_and_read_json_file(f_pipeline_paths)
    f_absolu_include_pipeline_json = os.path.join(pipeline_paths["pipelines"], f_relative_include_pipeline_json)
    include_pipeline        = check_exists_and_read_json_file(f_absolu_include_pipeline_json)

    # 2) Exécuter le pipeline
    execute_pipeline(
        global_params = params,
        global_data = data,
        pipeline = include_pipeline,
        pipeline_name = f_relative_include_pipeline_json,
        verbose = True)

    if ("output" in include_pipeline.keys()):
        value = include_pipeline["output"]
        
        if (isinstance(value, list)):
            # value est en fait une liste
            value_list = value

            # Construction de la liste des arguments de retour
            output_list = []
            for value in value_list:
                if (value.startswith("param_")):
                    output_val = get_param(params, value)

            
                if (value.startswith("data_")):
                    output_val = get_data(data, value)
                output_list.append(output_val)
            return output_list

        if (isinstance(value, str)):
            if (value.startswith("param_")):
                return get_param(params, value)
        
            if (value.startswith("data_")):
                return get_data(data, value)

    return


def recupere_valeur_variable(global_params, params_for_paths_search, variable):
    for path in params_for_paths_search:
        if (path == "/"):
            if (variable in global_params.keys()):
                return global_params[variable]
        else:
            if (variable in global_params[path].keys()):
                return global_params[path][variable]
    print("main_pipeline > Erreur: Variable non trouvée dans les paramètres:", variable)
    exit(-1)
    return

def parse_dict_name_and_key(value):
    if (type(value) == str):
        p = re.compile("^\$([a-zA-Z0-9_]+)\[(\$?[a-zA-Z0-9_]*)\]$")
        parse_res = p.findall(value)
        if (len(parse_res) == 1):
            if (len(parse_res[0]) == 2):
                return parse_res[0]
    return None

def est_un_dict(value):
    return (parse_dict_name_and_key(value) != None)

def est_une_variable(value):
    if (type(value) == str):
        if (value.startswith("$")):
            return True
    elif (type(value) == list):
        for v in value:
            if (est_une_variable(v)):
                return True
    return False


def construit_valeur_variable(global_params, params_for_paths_search, value):
    if (type(value) == str):
        if (est_une_variable(value)):
            b_dict = est_un_dict(value)
            if (b_dict):
                [dict_name, dict_key] = parse_dict_name_and_key(value)
                my_dict = recupere_valeur_variable(global_params, params_for_paths_search, dict_name)
                my_key  = construit_valeur_variable(global_params, params_for_paths_search, dict_key)
                if (my_key in my_dict.keys()):
                    return construit_valeur_variable(global_params, params_for_paths_search, my_dict[my_key])
                else:
                    print("Erreur, clef '" + my_key + "' non trouvée dans le dictionnaire. Clefs disponibles:")
                    print(my_dict.keys())
                    exit(-1)
                return None
            else:
                variable = value.replace("$", "")
                valvar = recupere_valeur_variable(global_params, params_for_paths_search, variable)
            return construit_valeur_variable(global_params, params_for_paths_search, valvar)
    elif (type(value) == list):
        valvar_list = ""
        for v in value:
            if (est_une_variable(v)):
                valeur = construit_valeur_variable(global_params, params_for_paths_search, v)
            else:
                valeur = v
            valvar_list = valvar_list + str(valeur)
        return valvar_list

    return value


# Fonction récursive qui effectue les remplacements des variables avec dollar
def get_param(global_params, param_name, verbose = False):
    if (not (param_name.startswith("param_"))):
        return None

    param_name  = re.sub("param_", "", param_name)
    param_value = global_params[param_name]
    if (est_une_variable(param_value)):
        if (not ("params_for_paths_search" in global_params.keys())):
            print("Error: The following parameter is missing within the param json file: params_for_paths_search.")
            exit(-1)

        params_for_paths_search = global_params["params_for_paths_search"]
        param_value = construit_valeur_variable(global_params, params_for_paths_search, param_value)
    #if (verbose):
    #    print("param: ", param_name, " = ", param_value)
    return param_value


def save_param(global_params, param_name, param_value, verbose = False):
    if (param_name.startswith("param_")):
        param_name = re.sub("param_", "", param_name)
        if (verbose):
            print("param_name:", param_name)
        global_params[param_name] = param_value



def get_data(global_data, data_name, verbose = False):
    if (data_name.startswith("data_")):
        data_name = re.sub("data_", "", data_name)
        if (verbose):
            print("Donnée:", data_name)
        return global_data[data_name]



def save_data(global_data, data_name, data_value, verbose = False):
    if (data_name.startswith("data_")):
        data_name = re.sub("data_", "", data_name)
        if (verbose):
            print("Donnée:", data_name)
        global_data[data_name] = data_value


def print_timestamp():
    time1 = str(datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    time2 = str(time.time())
    print("timestamp ; " + time1 + " ; " + time2)


def print_debut_fin_pipeline(true_debut__false_fin, pipeline_name, global_params = None):
    debutfin = "Fin"
    if (true_debut__false_fin):
        debutfin = "Debut"
    
    print(" ")
    print("########################")
    if (len(pipeline_name) > 0):
        print(debutfin + " de execute_pipeline() avec pipeline_name = ", pipeline_name)
    else:
        print(debutfin + " de execute_pipeline()")
    print_timestamp()
    print("########################")
    if (true_debut__false_fin):
        print("Paramètres globaux du pipeline:")
        print(json.dumps(global_params, indent=2))
        print("########################")

    print(" ")
    sys.stdout.flush()

def execute_pipeline(global_params, global_data, pipeline, pipeline_name = "", verbose = False):
    if (verbose):
        print_debut_fin_pipeline(
            true_debut__false_fin = True,
            pipeline_name = pipeline_name,
            global_params = global_params)

    if ("provides" in pipeline.keys()):
        f_provides = get_param(global_params, pipeline["provides"], verbose)
        #print("debug provides:", f_provides)
        if (os.path.exists(f_provides)):
            print("provides file exists:", f_provides)
            print("lazy pipeline ended")
            if (verbose):
                print_debut_fin_pipeline(
                    true_debut__false_fin = False,
                    pipeline_name = pipeline_name)


            return

    for module in pipeline["modules"]:

        if ("name" in module.keys()):
            module_name = module["name"]
            #if (True):
            if (verbose):
                print(" ")
                print("Module:", module_name)
                sys.stdout.flush()

            # Chargement des arguments du module 
            local_params = {}
            local_data = {}
            for key, value in module.items():
                if (key != "output"):
                    #print(key, " = ", value)
                    if (verbose):
                        print("* value:", value)
                    if (type(value) == dict):
                        local_params[key] = value
                    elif (type(value) == str):
                        if (value.startswith("param_")):
                            local_params[key] = get_param(global_params, value, verbose)
                        elif (value.startswith("data_")):
                            local_data[key] = get_data(global_data, value, verbose)
                        elif (value.startswith("internal_")):
                            local_params[key] = re.sub("internal_", "", value)

                
            # Appel au module
            #module_output = 1
            if (verbose):
                print("Appel à la fonction du module:", module_name)
                print_timestamp()
                sys.stdout.flush()

            if (module_name == "module_include"):
                module_output = module_include(local_params, local_data)
            else:
                # Generic function pipeline_interface.py
                module_output = module_pipeline_generic_function(module_name, local_params, local_data)
                #print("type module_output:", type(module_output))

            # Récupération de la sortie du module
            if ("output" in module.keys()):
                value = module["output"]
                if (verbose):
                    print("* output value:", value)

                if (isinstance(value, list)):
                    # value est en fait une liste
                    value_list = value
                    module_output_list = module_output

                    # Parcours de la liste
                    output_index = 0
                    for value in value_list:
                        module_output = module_output_list[output_index]
                        print("nouvelle iteration:", output_index, value, module_output)
                        print("effectue interation")
                        if (value.startswith("param_")):
                            save_param(global_params, value, module_output, verbose)
                
                        if (value.startswith("data_")):
                            save_data(global_data, value, module_output, verbose)

                        output_index = output_index + 1

                if (isinstance(value, str)):                    
                    if (value.startswith("param_")):
                        save_param(global_params, value, module_output, verbose)
            
                    if (value.startswith("data_")):
                        save_data(global_data, value, module_output, verbose)

            #print(global_params)
            #print("global_data:", global_data)


    if (verbose):
        print_debut_fin_pipeline(
            true_debut__false_fin = False,
            pipeline_name = pipeline_name)



########################

def check_exists_and_read_json_file(f_path):
    #print("debug(1):", f_path)
    if (os.path.exists(f_path)):
        with open(f_path, 'r') as j:
            print("Lecture du fichier json:", f_path)
            return json.loads(j.read())
    else:
        print("Erreur: fichier non trouvé (" + f_path + ")")
        exit(-1)        
        return None

def load_pipeline_parameters(parameters_folder, f_params_path):
    f_full_path        = os.path.join(parameters_folder, f_params_path)
    pipeline_parameters = check_exists_and_read_json_file(f_full_path)

    output_pipeline_parameters = {}
    # Parsing du fichier de paramètres, à la recherche d ' "include_params"
    # On insère d'abord les paramètres des fichiers d'inclusion (valeurs par défaut)
    for key, value in pipeline_parameters.items():
        if (key.startswith("include_params")):
            print("Inclusion du fichier de paramètres:", value)
            parameters_from_include_curr = load_pipeline_parameters(parameters_folder, value)
            output_pipeline_parameters.update(parameters_from_include_curr)

    # Puis on insère les paramètres du fichier principal
    # Ces paramètres ont la priorité par rapport aux paramètres des fichiers inclus
    output_pipeline_parameters.update(pipeline_parameters)

    return output_pipeline_parameters



def load_file_params_surcharge_pipeline_repetition(parameters_folder, f_params_pspr_path):
    PSPR = "params_surcharge_pipeline_repetition"
    f_full_path        = os.path.join(parameters_folder, f_params_pspr_path)
    pspr_parameters    = check_exists_and_read_json_file(f_full_path)

    # Vérifier la conformité du fichier pspr (params_surcharge_pipeline_repetition)
    if (not (PSPR in pspr_parameters.keys())):
        print("Erreur (1): Le fichier pspr n'est pas conforme (" + f_full_path + ")")
        print("La seule clef à la racine du json doit être :" + PSPR)
        exit(-1)
    for key, value in pspr_parameters.items():
        if (key != PSPR):
            print("Erreur (2): Le fichier pspr n'est pas conforme (" + f_full_path + ")")
            print("La seule clef à la racine du json doit être :" + PSPR)
            exit(-1)

    params_pspr = pspr_parameters[PSPR]
    n_runs = len(params_pspr)

    # On parcoure/parse toutes les clefs du dictionnaire de paramètres de runs
    # pour vérifier leur conformité (Nécessaire pour pouvoir lancer les runs dans l'ordre,
    # potentiellement pouvoir relancer seulement certains d'entre eux)
    for index_run in range(n_runs):
        key_run = "run_" + str(index_run + 1)
        if (not (key_run in params_pspr.keys())):
            print("Erreur (3): Le fichier pspr n'est pas conforme (" + f_full_path + "). Clef manquante:", key_run)
            print("Les clefs du dictionnaire de paramètres de run doivent être sous la forme 'run_$index', par exemple 'run_4', pour le 4eme run")
            exit(-1)

    return params_pspr         



# Code principal

print(" ")
print("Début de main_pipeline.py")

n_params_pspr = 0
if (len(sys.argv) >= 4):
    n_params_pspr = len(sys.argv) - 4
else:
    print("Incorrect number of arguments for " + sys.argv[0])
    print("Need at least 3 arguments: pipeline.json, params.json, logfile")
    exit(-1)

f_pipeline = sys.argv[1]
f_params   = sys.argv[2]
f_logs     = sys.argv[3]
f_params_pspr_path_list = sys.argv[4:(4 + n_params_pspr)]

print("Fichier de pipeline  :", f_pipeline)
print("Fichier de paramètres:", f_params)
print("Fichier de logs      :", f_logs)
print("Fichiers pspr     :", f_params_pspr_path_list)

# Tester l'accés aux fichiers puis lancer le pipeline
pipeline_paths       = check_exists_and_read_json_file(f_pipeline_paths)

f_path_pipeline      = os.path.join(pipeline_paths["pipelines"], f_pipeline)
pipeline_structure   = check_exists_and_read_json_file(f_path_pipeline)

pipeline_parameters  = load_pipeline_parameters(pipeline_paths["parameters"], f_params)

params_pspr_list = []
for f_params_pspr_path in f_params_pspr_path_list:
    params_pspr_list.append(load_file_params_surcharge_pipeline_repetition(pipeline_paths["parameters"], f_params_pspr_path))


if (not (os.path.exists(pipeline_paths["logs"]))):
    print("Erreur: Le dossier de logs n'existe pas")
    exit(-1)

f_path_logs          = os.path.join(pipeline_paths["logs"], f_logs)

if (os.path.exists(f_path_logs)):
    print("Warning: Le fichier de logs existe déjà.")


# Les fichiers et dossiers principaux du pipeline existent
# et ne sont pas corrompus:
# On redirige la sortie standard vers le fichier de logs,
# et on lance le pipeline

f = open(f_path_logs, 'a')
sys.stdout = f

print(" ")
print("################################################")
print("Début de logging du pipeline général")
print("################################################")
print(" ")

print("Fichier de pipeline  :", f_pipeline)
print("Fichier de paramètres:", f_params)
print("Fichier de logs      :", f_logs)
if (n_params_pspr):
    print("Fichier pspr     :", f_params_pspr_path)


# Un seul run de pipeline, appel classique
if (len(params_pspr_list) == 0):

    # Lancement effectif du pipeline: Début de la partie "deep learning"
    global_data = {}
    execute_pipeline(pipeline_parameters, global_data, pipeline_structure, pipeline_name = "appel unique depuis main_pipeline.py", verbose = verbose)

elif (len(params_pspr_list) == 1):
    # Cas avec un seul fichier pspr fourni
    params_pspr = params_pspr_list[0]

    # Multiples lancements du pipeline,
    # avec des jeux de paramètres surchargés/modifiés par le pspr
    # On s'assure d'effectuer les runs dans l'ordre
    n_runs = len(params_pspr)
    for index_run in range(n_runs):
        key_run = "run_" + str(index_run + 1)
        params_pspr_cour = params_pspr[key_run]
        # (Re)Chargement du fichier général de paramètres => valeurs par défaut
        pipeline_parameters  = load_pipeline_parameters(pipeline_paths["parameters"], f_params)
        # Mise à jour/surcharge des paramètres spécifiés dans le fichier pspr
        pipeline_parameters.update(params_pspr_cour)        

        # Lancement effectif du pipeline: Début de la partie "deep learning"
        global_data = {}
        execute_pipeline(pipeline_parameters, global_data, pipeline_structure, pipeline_name = key_run + " de " + f_params_pspr_path, verbose = verbose)
else:
    # Cas avec plusieurs fichiers fournis: pas encore implémenté
    print("Cas avec plusieurs fichiers pspr pas encore implémenté.")
    exit(-1)

print(" ")
print("################################################")
print("Fin de logging du pipeline général")
print("################################################")
print(" ")


# Remettre la bonne sortie standard
sys.stdout = sys.__stdout__

