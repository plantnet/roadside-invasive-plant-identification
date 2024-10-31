#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""This module provides basic code to compare tensors by displaying their signature/caracteristic (shape, and stats over values)"

Author: Vincent Espitalier <vincent.espitalier@cirad.fr>
        Hervé Goëau <herve.goeau@cirad.fr>
"""

# $ micromamba activate pytorch2
# $ python util_diff_tensors.py
# $ python util_diff_tensors.py tensor1.pth tensor2.pth

EPS = 0.0000001

import sys
import os
import torch

if (False):

    if (len(sys.argv) != 3):
        print("Erreur: nécessite 2 arguments (2 noms de fichier)")
        exit(-1)

    #f_in_1 = sys.argv[1]
    #f_in_2 = sys.argv[2]

    f_in_reference = "../../0_datastore/20_deep_features/2020-08-17_verif_09-10/GT_2020-08-17T08_45_06.000Z_CT_1597329574.5417304_10.037836_56.1166185.jpg_reference.pth"

    f_in_cls_sans_autocast = "../../0_datastore/20_deep_features/2020-08-17_verif_09-10/GT_2020-08-17T08_45_06.000Z_CT_1597329574.5417304_10.037836_56.1166185.jpg_cls_sans_autocast.pth"
    f_in_moy_sans_autocast = "../../0_datastore/20_deep_features/2020-08-17_verif_09-10/GT_2020-08-17T08_45_06.000Z_CT_1597329574.5417304_10.037836_56.1166185.jpg_moy_sans_autocast.pth"

    f_in_cls_autocast = "../../0_datastore/20_deep_features/2020-08-17_verif_09-10/GT_2020-08-17T08_45_06.000Z_CT_1597329574.5417304_10.037836_56.1166185.jpg_cls_autocast.pth"
    f_in_moy_autocast = "../../0_datastore/20_deep_features/2020-08-17_verif_09-10/GT_2020-08-17T08_45_06.000Z_CT_1597329574.5417304_10.037836_56.1166185.jpg_moy_autocast.pth"

    f_in_old = "../../0_datastore/20_deep_features/2020-08-17_verif_09-10/GT_2020-08-17T08_45_06.000Z_CT_1597329574.5417304_10.037836_56.1166185.jpg_old.pth"

    f_in_moy_sans_autocast_1010 = "../../0_datastore/20_deep_features/2020-08-17_verif_09-10/GT_2020-08-17T08_45_06.000Z_CT_1597329574.5417304_10.037836_56.1166185.jpg_moy_sans_autocast_10-10.pth"


    #f_in_1 = f_in_cls_sans_autocast
    #f_in_1 = f_in_moy_sans_autocast
    #f_in_1 = f_in_cls_autocast
    #f_in_1 = f_in_moy_autocast
    #f_in_1 = f_in_reference
    f_in_1 = f_in_old

    #f_in_2 = f_in_cls_sans_autocast
    #f_in_2 = f_in_moy_sans_autocast
    f_in_2 = f_in_moy_sans_autocast_1010
    #f_in_2 = f_in_cls_autocast
    #f_in_2 = f_in_moy_autocast
    #f_in_2 = f_in_reference


    if (not os.path.exists(f_in_1)):
        print("Le fichier n'existe pas: ", f_in_1)
        exit(-1)

    if (not os.path.exists(f_in_2)):
        print("Le fichier n'existe pas: ", f_in_2)
        exit(-1)

    try:
        t_in_1 = torch.load(f_in_1)
        print("Tenseur chargé:", f_in_1)
    except:
        print("Le fichier ne peut etre chargé dans pytorch: ", f_in_1)
        exit(-1)
        
    try:
        t_in_2 = torch.load(f_in_2)
        print("Tenseur chargé:", f_in_2)
    except:
        print("Le fichier ne peut etre chargé dans pytorch: ", f_in_2)
        exit(-1)
        
    if (not torch.is_tensor(t_in_1)):
        print("Le fichier n'est pas un tenseur: ", f_in_1)
        exit(-1)
        
    if (not torch.is_tensor(t_in_2)):
        print("Le fichier n'est pas un tenseur: ", f_in_2)
        exit(-1)
        
    if (t_in_1.shape != t_in_2.shape):
        print("Les tenseurs n'ont pas la meme shape: ", f_in_1, f_in_2)
        exit(-1)


    t_relativ = torch.div( (t_in_2 - t_in_1) , 0.5 * (torch.abs(t_in_2) + torch.abs(t_in_1) + EPS) )

    print("Shape des tenseurs:", t_in_1.shape)
    print(t_in_1)

    #print(t_in_1[920:925, 200:210])

    print("Ecart type de la difference relative entre tenseurs:", int(10000. * torch.std(t_relativ).item()) / 100., "% ")

    print(" ")
    print("Moy tenseur1 : ", torch.mean(t_in_1).item())
    print("Moy tenseur2 : ", torch.mean(t_in_2).item())
    std1 = torch.std(t_in_1).item()
    std2 = torch.std(t_in_2).item()
    print("Ecart type tenseur1 : ", std1)
    print("Ecart type tenseur2 : ", std2)
    std_moy = 0.5 * (std1 + std2)
    d_max = torch.max(torch.abs(t_in_2 - t_in_1)).item()
    print("Difference maximale entre tenseurs:", d_max, " ; ", int(10000. * d_max / std_moy) / 100., "% d'ecart type moyen")

    std_diff = torch.std(t_in_2 - t_in_1).item()
    print("Ecart type de la difference entre tenseurs:", int(10000. * std_diff / std_moy) / 100., "% d'ecart type moyen")



###########

if (True):
    # Comparer tous les tenseurs d'un dossier, par leur "signature"

    f_in_list = []
    #f_folder = "/home/vincent/Bureau/Bureau2/4_pipeline_quadrats_officiel_danois/0_datastore/20_deep_features/verif_14-10/"
    #f_folder = "../../0_datastore/20_deep_features/DanishRoads/"
    f_folder = "../../0_datastore/20_deep_features/DanishRoads/mads_split/test_14_new/1_solidago/"
    
    #f_folder = "./"

    # Recherche dans les dossiers et sous-dossiers
    #f_in_list = [f_in for f_in in os.listdir(f_folder) if (".pth" in f_in)]
    f_in_list = [os.path.join(os.path.relpath(root, f_folder), file) for root, dirs, files in os.walk(f_folder) for file in files if (".pth" in file)]
    f_in_list = list(sorted(f_in_list, key = str))

    # Afficher une "signature" de chaque tenseur, pour comparer plus de 2 tenseurs
    print(" ")
    print("Signature liste de tenseurs:")
    for f_in in f_in_list:
        f_full_in = os.path.join(f_folder, f_in)
        t_in = torch.load(f_full_in)
        t_in_abs = torch.abs(t_in)

        t_in_flat = t_in.flatten().to(torch.device("cpu"))
        indices = torch.arange(t_in_flat.size(0)).to(torch.device("cpu")) / 1000.
        code_control = t_in_flat * indices
        
        #f_short = f_in[-30:]
        f_short = f_in.split("/")[-1]

        print(f_short, ";", list(t_in.shape), ";", torch.mean(t_in).item(), ";", torch.std(t_in).item(), ";", torch.mean(t_in_abs).item(), ";",  torch.std(t_in_abs).item(), ";", torch.mean(code_control).item(), ";", torch.std(code_control).item())
    print(" ")
