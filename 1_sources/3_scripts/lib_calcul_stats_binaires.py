#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""This module provides code to compute class-wise and multilabel detection statistics

Author: Vincent Espitalier <vincent.espitalier@cirad.fr>
        Hervé Goëau <herve.goeau@cirad.fr>
"""

import re
import os 
import torch
import math
import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset


from lib_utils_files import charge_fichier_csv_vers_dataframe, enregistre_dataframe_vers_fichier_csv_excel, create_folder_if_does_not_exists, write_to_json_file, read_json_file, pretty_json

from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, \
    accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, auc, jaccard_score

from sklearn.linear_model import LinearRegression


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def charge_annotations_csv(input_df_annotations, input_df_predictions):
    # Allow empty fields for annotations
    df_all_annotations = input_df_annotations.replace(np.nan, 0)

    # Tableau avec les meme colonnes, vide.
    #df_annotations = pd.DataFrame().reindex_like(df_all_annotations)
    df_annotations = df_all_annotations[0:0]
    
    #print("df_all_annotations:", df_all_annotations)
    #print("df_annotations:", df_annotations)
    #exit()
    
    index_out = 0
    for index in range(len(input_df_predictions["image_name"])):
        f_image_name = input_df_predictions["image_name"][index]
        f_image_name = f_image_name.replace(".jpg", "")
        if ((index % 100) == 0):
            print(index, "'" + f_image_name + "'")

        index_in = df_all_annotations["image_name"].str.contains(f_image_name)
        n_match = df_all_annotations[index_in].shape[0]
        if (n_match != 1):
            print("Error: Mismatch in images names between annotations and predictions")
            print("f_image_name:", f_image_name, "; n_match:", n_match, "; index_in:", index_in)
            exit(-1)
        #print("df_all_annotations[index_in]:", df_all_annotations[index_in].values)
        #df_annotations.loc[-1] = df_all_annotations[index_in]
        #df_annotations.loc[-1] = df_all_annotations[index_in].values[0]
        #df_annotations.loc[-1] = df_all_annotations[index_in].values
        #df_annotations.loc[-1] = df_all_annotations[index_in][0]
        df_annotations = pd.concat([df_annotations, df_all_annotations[index_in] ])
        index_out = index_out + 1

    return df_annotations

def derive_poly(coefs_poly):
    poly_deg  = coefs_poly.size - 1
    if (poly_deg < 0):
        return np.zeros(0)
    deriv_deg = poly_deg - 1
    deriv     = np.zeros(deriv_deg + 1)
    for i in range(deriv_deg + 1):
        deriv[i] = (i + 1) * coefs_poly[i + 1] 
    return deriv


def eval_poly(x, coefs_poly):
    poly_deg  = coefs_poly.size - 1
    if (poly_deg < 0):
        return 0.0
    y = 0.
    # Méthode de calcul naïve
    for i in range(poly_deg + 1):
        y = y + coefs_poly[i] * (x ** i)
    return y



def normal_density(x, mean = 0.0, var = 1.0):
    if (var > 0):
        x_m = x - mean
        facteur_1 = (1.0 / math.sqrt(2 * math.pi * var))
        facteur_2 = math.exp(-0.5 * x_m * x_m / var)
        return facteur_1 * facteur_2
    else:
        if (x == mean):
            return float('inf')
        else:
            return 0.0



# Régression linéaire avec scikit-learn
def scikitlearn_linear_regression(x_train, y_train):
    #print("x_train.shape:", x_train.shape)
    #X = x_train[:, 1:].cpu().detach().numpy()
    X = x_train.cpu().detach().numpy()
    y = y_train.cpu().detach().numpy()
    reg = LinearRegression(fit_intercept = False).fit(X, y)

    retour = {}

    retour["R"] = reg.score(X, y)
    retour["coefficients"] = reg.coef_
    return retour



def calcule_stats_et_densite(predicted, use_log, polynom_degres, device):
    if ((polynom_degres % 2) != 1):
        print("Erreur: Le polynôme doit être monotone donc de degré impair (degré:", polynom_degres, " fourni)")
        exit(-1)

    if (predicted.shape[0] == 0):
        print("Erreur: calcule_stats_et_densite > tenseur 'predicted' vide: L'espece est-elle bien présente dans le dataset ?")
        exit(-1)
    
    stats_densite = {}
    stats_densite["mean"]    = torch.mean(predicted).item()
    stats_densite["std"]     = torch.std(predicted).item()
    stats_densite["median"]  = torch.median(predicted).item()
    stats_densite["min"]     = torch.min(predicted).item()
    stats_densite["max"]     = torch.max(predicted).item()

    stats_densite["q0.0"]    = torch.quantile(input = predicted, q = 0.0).item()
    stats_densite["q0.005"]  = torch.quantile(input = predicted, q = 0.005).item()
    stats_densite["q0.01"]   = torch.quantile(input = predicted, q = 0.01).item()
    stats_densite["q0.05"]   = torch.quantile(input = predicted, q = 0.05).item()
    stats_densite["q0.1"]    = torch.quantile(input = predicted, q = 0.1).item()
    stats_densite["q0.25"]   = torch.quantile(input = predicted, q = 0.25).item()
    stats_densite["q0.5"]    = torch.quantile(input = predicted, q = 0.5).item()
    stats_densite["q0.75"]   = torch.quantile(input = predicted, q = 0.75).item()
    stats_densite["q0.9"]    = torch.quantile(input = predicted, q = 0.9).item()
    stats_densite["q0.95"]   = torch.quantile(input = predicted, q = 0.95).item()
    stats_densite["q0.99"]   = torch.quantile(input = predicted, q = 0.99).item()
    stats_densite["q0.995"]  = torch.quantile(input = predicted, q = 0.995).item()
    stats_densite["q1.0"]    = torch.quantile(input = predicted, q = 1.0).item()

    # 0.5%, 1%, 5%, 10%, 25%, 50%, 75%, 90%, 95%, 99%, 99.5%

    if (polynom_degres > 0):

        # 1) Trier est entrées
        sorted_predicted, _ = predicted.sort()

        # Appliquer le log si demandé
        if (use_log):
            reglin_x = torch.log(sorted_predicted)
        else:
            reglin_x = sorted_predicted


        n_predicted = sorted_predicted.shape[0]

        # 2) Fabriquer des vecteurs de la même taille contenant les lois normales cumulées inverse
        torch_range = torch.arange(start = 1, end = n_predicted + 1, step = 1, dtype = torch.float32, device = device)
        quantile    = torch.div(torch_range, float(n_predicted + 1))
        normicdf    = torch.distributions.Normal(0, 1).icdf(quantile)
        #print(quantile)
        #print("normicdf:", normicdf.shape)
        #print("torch_range:", torch_range.shape)
        #exit()

        # 3) Régression linéaire polynome Q:  normicdf = Q(reglin_x)
        # Degré du polynôme: polynom_degres

        # 3a) On fabrique un tenseur de taille (n_present, polynom_degres + 1)
        # Chaque colonne représente un monôme du polynôme

        # Calcul pas optimisé mais suffisant pour de petites tailles (préférer Horner)
        reglin_X_pows = torch.zeros(n_predicted, polynom_degres + 1, device = device)
        for i in range(polynom_degres + 1):
            #print(reglin_X_pows[:, i].shape)
            #print(torch.pow(reglin_x, i)[0].shape)
            reglin_X_pows[:, i] = torch.pow(reglin_x, i)

        # 3b) Ensuite on fait une régression multilinéaire
        #X = reglin_X_pows
        #Y = normicdf
        lr_result = scikitlearn_linear_regression(
                        x_train = reglin_X_pows,
                        y_train = normicdf)

        # Ne pas flooder le fichier json de stats des logits
        #stats_densite.update(lr_result)
        stats_densite["R"] = lr_result["R"]

        stats_densite["degres"] = polynom_degres

        #print(type(lr_result["coefficients"]))


        coefs_poly = lr_result["coefficients"]
        #print("Polynome:", coefs_poly)
        deriv_poly = derive_poly(coefs_poly)
        #print("Polynome dérivé:", deriv_poly)

        # 4) Vérifier que le polynome dérivé Q' est strictement positif:
        # 5) en calculant la densité pour chaque point du predicted
        list_densite = []
        cumul_check = 0.0
        for index_x_cour_val in range(sorted_predicted.shape[0]):
            x_cour = sorted_predicted[index_x_cour_val].item()
            if (use_log):
                if (x_cour <= 0.):
                    facteur_1 = 0.0
                    facteur_2 = 0.0
                    facteur_3 = 0.0
                else:
                    log_x_cour = math.log(x_cour)
                    facteur_1 = (1 / x_cour)
                    facteur_2 =                eval_poly(log_x_cour, deriv_poly)
                    facteur_3 = normal_density(eval_poly(log_x_cour, coefs_poly))
            else:
                facteur_1 =                eval_poly(x_cour, deriv_poly)
                facteur_2 = normal_density(eval_poly(x_cour, coefs_poly))
                facteur_3 = 1.0

            densite_cour = facteur_1 * facteur_2 * facteur_3

            if (densite_cour < 0):
                #print("densite negative pour ", x_cour)
                #print(facteur_1, facteur_2, facteur_3)
                new_polynom_degres = polynom_degres - 2
                return calcule_stats_et_densite(predicted, use_log, new_polynom_degres, device)
            list_densite.append([x_cour, densite_cour])
            if (index_x_cour_val > 0):
                cumul_check = cumul_check + densite_cour * (sorted_predicted[index_x_cour_val].item() - sorted_predicted[index_x_cour_val - 1].item())

        stats_densite["densite"] = list_densite
        stats_densite["cumul_check"] = cumul_check

    # 6) Renvoyer la densité + R2 + les coefficients du polynôme pour réutilisation
    return stats_densite



def calcule_densites(labels, predicted, use_log, polynom_degres, device):
    # labels, predicted : Pytorch tenseurs

    # Séparer le tenseur de predicted en 2 parties: labels = 1 ou 0
    # pour calculer 2 densités.

    print("labels:", labels)
    print("predicted:", predicted)
    print("predicted.shape:", predicted.shape)
    print("unique labels:", torch.unique(labels))

    print("indexes")
    indexes_present = (labels == 1).nonzero()
    indexes_absent  = (labels == 0).nonzero()
    #print (indexes_present)
    #print (indexes_absent)
    n_present = indexes_present.shape[0]
    n_absent  = indexes_absent.shape[0]
    print (n_present)
    print (n_absent)

    print("predicted_present")
    predicted_present = predicted[indexes_present][:, 0]
    #print (predicted_present)
    #print (predicted_present.shape)
    #exit()

    print("predicted_absent")
    predicted_absent  = predicted[indexes_absent][:, 0]
    #print (predicted_absent)
    #print (predicted_absent.shape)

    print("predicted_tous")
    predicted_tous  = predicted
    #print (predicted_absent)
    #print (predicted_absent.shape)

    stats_present = calcule_stats_et_densite(predicted_present, use_log, polynom_degres, device)
    stats_absent  = calcule_stats_et_densite(predicted_absent, use_log, polynom_degres, device)
    stats_tous    = calcule_stats_et_densite(predicted_tous, use_log, polynom_degres, device)

    print ("moyenne de proba pred quand présent:", stats_present["mean"])
    print ("moyenne de proba pred quand absent:",  stats_absent["mean"])

    print ("ecart-type de proba pred quand présent:", stats_present["std"])
    print ("ecart-type de proba pred quand absent:",  stats_absent["std"])


    return {"stats_present" : stats_present, "stats_absent" : stats_absent, "stats_tous" : stats_tous}



def plot_stats_distrib_proba(stats, classe, scale_min_max, f_output_densite, proba_threshold):
    densite_present = stats["stats_present"]["densite"]
    densite_absent  = stats["stats_absent"]["densite"]
    R2_present = stats["stats_present"]["R"]
    R2_absent = stats["stats_absent"]["R"]
    degres_present = stats["stats_present"]["degres"]
    degres_absent  = stats["stats_absent"]["degres"]
    cumul_check_present = stats["stats_present"]["cumul_check"]
    cumul_check_absent  = stats["stats_absent"]["cumul_check"]
    mean_present = stats["stats_present"]["mean"]
    mean_absent  = stats["stats_absent"]["mean"]

    fig, ax = pyplot.subplots()

    x_present = []
    y_present = []
    for index in range(len(densite_present)):
        x_present.append(densite_present[index][0])
        y_present.append(densite_present[index][1])

    max_y_present = max(y_present)

    if ("min_x" in scale_min_max):
        min_x = scale_min_max["min_x"]
        pyplot.xlim(left  = min_x)

    if ("max_x" in scale_min_max):
        max_x = scale_min_max["max_x"]
        pyplot.xlim(right  = max_x)

    if ("min_y" in scale_min_max):
        min_y = scale_min_max["min_y"]
        pyplot.ylim(bottom = min_y)

    if ("max_y" in scale_min_max):
        max_y = scale_min_max["max_y"]
        pyplot.ylim(top = max_y)
        
    pyplot.plot(x_present, y_present, marker='.', label='present', color="green")

    #pyplot.xscale("log")
    pyplot.xscale("linear")     # TODO:  Exporter ce flag pour éviter ce hack

    #pyplot.yscale("log")
    pyplot.yscale("linear")
    
    pyplot.xlabel('Logit')
    #pyplot.ylabel('Density is present')
    pyplot.legend()

    #f_output_image = "densite_present.png"
    #pyplot.savefig(f_output_image)

    x_absent = []
    y_absent = []
    for index in range(len(densite_absent)):
        x_absent.append(densite_absent[index][0])
        y_absent.append(densite_absent[index][1])

    max_y_absent = max(y_absent)

    pyplot.plot(x_absent, y_absent, marker='.', label='absent', color="orange")
    #pyplot.xlim(left  = min_x)
    #pyplot.xlim(right = max_x)
    #pyplot.ylim(top = max_y_graph)
    
    #pyplot.xscale("log")
    pyplot.xscale("linear")
    
    #pyplot.yscale("log")
    pyplot.yscale("linear")
    
    pyplot.xlabel('Logit')
    #pyplot.ylabel('Density if absent')

    def str_nb(nb, prec = 4):
        return str(int(nb * (10 ** prec)) / (10 ** prec))

    # Ajout du texte
    n_present = len(x_present)
    n_absent  = len(x_absent)
    s_specie = classe
    s_present = "present (" + str(n_present) + ")" \
            + "; mean = " + str_nb(mean_present) \
            + ": degres="  + str(degres_present) \
            + "; R2="     + str_nb(R2_present) \
            + "; cumul_check=" + str_nb(cumul_check_present)

    s_absent = "absent (" + str(n_absent) + ")" \
            + "; mean = " + str_nb(mean_absent) \
            + ": degres="  + str(degres_absent) \
            + "; R2="     + str_nb(R2_absent) \
            + "; cumul_check=" + str_nb(cumul_check_absent)

    pyplot.text(.001, .999, s_specie + "\n" + s_present + "\n" + s_absent, ha='left', va='bottom', transform=ax.transAxes)

    if (True):
        # Ajout de la ligne de seuil
        max_line_seuil = max(max(y_absent), max(y_present))
        #proba_threshold_label = "Max J Youden"
        proba_threshold_label = "Max J Youden sur val-set"
        pyplot.plot([proba_threshold, proba_threshold], [0.0, max_line_seuil], label=proba_threshold_label, color="blue")

        if (False):
            # Ligne de seuil pour matcher Mads
            model_choix_seuil_xp5h_match_recall_mads = {
                "Solidago"            : 0.1140,
                "Cytisus scoparius"   : 0.1430,
                "Rosa rugosa"         : 0.1350,
                "Lupinus polyphyllus" : 0.1900,
                "Pastinaca sativa"    : 0.1550,
                "Reynoutria"          : -0.0005}  

            proba_threshold_label = 'seuil_Mads'
            proba_threshold_value = model_choix_seuil_xp5h_match_recall_mads[classe]
            pyplot.plot([proba_threshold_value, proba_threshold_value], [0.0, max_line_seuil], label=proba_threshold_label, color="purple")

        if (False):
            # Ligne de seuil - Autre (Max F1 Score)
            model_choix_seuil_xp5c_Max_F1_Score = {
                "Solidago"            : 15.814487,
                "Cytisus scoparius"   : 9.652628,
                "Rosa rugosa"         : 9.446442,
                "Lupinus polyphyllus" : 8.2745905,
                "Pastinaca sativa"    : 10.450982,
                "Reynoutria"          : 29.301842}  

            proba_threshold_label = 'Max F1 Score'
            proba_threshold_value = model_choix_seuil_xp5c_Max_F1_Score[classe]
            pyplot.plot([proba_threshold_value, proba_threshold_value], [0.0, max_line_seuil], label=proba_threshold_label, color="purple")


    pyplot.legend()

    pyplot.savefig(f_output_densite)



# Internal function which calculates statistics for one class only (une specie)
# Input data are pandas dataFrames (df_annotations et df_predictions)
def calcule_stats_test_binaire_une_classe(
        df_annotations,
        df_predictions,
        f_out_path_images_prefixe,
        classe,
        model_choix_seuil,
        interp_densite):


    device = torch.device("cpu")

    labels    = torch.tensor(df_annotations[classe].values).to(device)

    predicted = torch.tensor(df_predictions[classe].values).to(device)


    n_instances = torch.sum(labels).item()
    print("Nombre d'instances pour la classe:", n_instances)

    n_images = df_predictions.shape[0]

    #print("debug")
    #print(labels.shape)
    #exit()
    #print(predicted)

    # Calculate the ROC curbes and precision-recall
    # as well as Youden's J and F-score statistics
    # ROC
    roc_fpr,  roc_tpr,  roc_thresholds  = roc_curve(labels, predicted)
    Youden_J = roc_tpr - roc_fpr

    # precision-recall
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(labels, predicted)
    pr_fscore = (2 * pr_precision * pr_recall) / (pr_precision + pr_recall)


    # Calculate the probability threshold following the value of "model_choix_seuil"
    if (isinstance(model_choix_seuil, dict)):
        proba_threshold = model_choix_seuil[classe]
    elif (isfloat(model_choix_seuil)):
        # A numerical value is provided -> use it
        proba_threshold = model_choix_seuil
    else:
        if (model_choix_seuil == "max_j_youden"):
            # Here we choose the threshold so as to maximize Youden's J statistics.
            # (It is relevant for ROC curves)
            # max(J Youden) = max(geometric average of sensibility * specificity)
            # Cf: https://machinelearningmastery.com/threshold-moving-for-imbalanced-classification/
            roc_index       = np.argmax(Youden_J)
            proba_threshold = roc_thresholds[roc_index]
            plot_dot_fpr    = roc_fpr[roc_index]
            plot_dot_tpr    = roc_tpr[roc_index]
        elif (model_choix_seuil == "max_f1_score"):
            # Here we choose the threshold so as to maximize f1 score aka balanced f-score
            # (relevant for precision-recall curves)
            pr_index = np.argmax(pr_fscore)
            proba_threshold = pr_thresholds[pr_index]
            plot_dot_recall = pr_recall[pr_index]
            plot_dot_precision = pr_precision[pr_index]


    if ("polynom_degres" in interp_densite):

        # Calcul des densités interpolées
        polynom_degres = interp_densite["polynom_degres"]
        use_log = interp_densite["use_log"]
        densites_stats = calcule_densites(labels, predicted, use_log, polynom_degres, device)
        #print(stats)

        if (polynom_degres > 0):
            # Grapher les densités interpolées
            scale_min_max = interp_densite["scale_min_max"]
            f_output_densite = os.path.join(f_out_path_images_prefixe, "densite_" + classe + ".png")
            plot_stats_distrib_proba(densites_stats, classe, scale_min_max, f_output_densite, proba_threshold)
            #R_present = densites_stats["stats_present"]["R"]
            #R_absent  = densites_stats["stats_absent"]["R"]
            print("R:", densites_stats["stats_present"]["R"], densites_stats["stats_absent"]["R"])
            print("degres:", densites_stats["stats_present"]["degres"], densites_stats["stats_absent"]["degres"])
            print("cumul_check:", densites_stats["stats_present"]["cumul_check"], densites_stats["stats_absent"]["cumul_check"])

        if ("output_stats_json" in interp_densite):
            if (interp_densite["output_stats_json"]):

                # ne pas flooder le fichier json de stats logits, avec toutes les densites
                densites_stats["stats_present"]["densite"] = []
                densites_stats["stats_absent"]["densite"] = []
                densites_stats["stats_tous"]["densite"] = []

                f_output_predicted_stats_json = os.path.join(f_out_path_images_prefixe, "predicted_stats_" + classe + ".json")
                #f_output_predicted_stats_json = os.path.join(f_out_path_images_prefixe, f_out_path_images_prefixe + "predicted_stats_" + classe + ".json")
                #print(densites_stats)
                #exit()
                write_to_json_file(f_output_predicted_stats_json, densites_stats)
        





    # structure returned by the function
    stat = {}

    stat["P"]        = n_instances
    stat["N"]        = n_images - n_instances
    stat["n_images"] = n_images

    output  = (predicted > proba_threshold).float()
    confusion_mat = confusion_matrix(labels, output)

    try:
        stat["TP"]      = confusion_mat[1][1]
        stat["FP"]      = confusion_mat[0][1]
        stat["TN"]      = confusion_mat[0][0]
        stat["FN"]      = confusion_mat[1][0]
    except:
        print("Erreurs dans le calcul de la matrice de confusion")
        print("Espèce:", classe)
        exit(-1)
    stat["correct"] = int((output == labels).float().sum().item())

    stat["accuracy"]          = accuracy_score(labels, output)
    stat["balanced_accuracy"] = balanced_accuracy_score(labels, output)
    stat["precision"]         = precision_score(labels, output)
    stat["recall"]            = recall_score(labels, output)
    if (stat["TN"] + stat["FP"] > 0):
        stat["specificite"]   = stat["TN"] / (stat["TN"] + stat["FP"])
        stat["FPR"]           = stat["FP"] / (stat["TN"] + stat["FP"])
    else:
        stat["specificite"]   = 0.0
        stat["FPR"]           = 0.0

    if (stat["TN"] + stat["FN"] > 0):
        stat["VPN"]           = stat["TN"] / (stat["TN"] + stat["FN"])
    else:
        stat["VPN"]           = 0.0


    stat["F1"]      = f1_score(labels, output)
    stat["TPR"]     = stat["TP"] / (stat["TP"] + stat["FN"])
    stat["FPR"]     = stat["FP"] / (stat["FP"] + stat["TN"])
    stat["Gmeans"]  = np.sqrt(stat["TPR"] * (1. - stat["FPR"]))

    stat["AUC"]     = auc(roc_fpr, roc_tpr)

    stat["proba_threshold"]     = float(proba_threshold)

    if (f_out_path_images_prefixe != ""):
        fig, ax = pyplot.subplots()

        # Plot the ROC curve and show the threshold
        #pyplot.plot([0,1], [0,1], linestyle='--', label='No Skill')
        #pyplot.plot(roc_fpr, roc_tpr, marker='.', label='Logistic')
        #pyplot.plot(roc_fpr, roc_tpr, label='ROC curve')
        #if (model_choix_seuil == "max_j_youden"):
        #pyplot.scatter(plot_dot_fpr, plot_dot_tpr, marker='o', color='black', label='Best')
        #pyplot.scatter(plot_dot_fpr, plot_dot_tpr, marker='o', color='black', label='Best')
        ax.plot(roc_fpr, roc_tpr, label='ROC curve')
        #ax.scatter(plot_dot_fpr, plot_dot_tpr, marker='o', color='black', label='Best')

        pyplot.xlabel('False Positive Rate')
        pyplot.ylabel('True Positive Rate')
        pyplot.legend()

        # Resserrer la grille
        #ax.xaxis.set_major_locator(pyplot.MultipleLocator(0.1))
        #ax.yaxis.set_major_locator(pyplot.MultipleLocator(0.1))

        #pyplot.grid(color = 'green', linestyle = '--', which='major', linewidth = 0.5)
        #pyplot.show()
        

        # Ajout de la grille supplémentaire avec axes
        #inset_axes = zoomed_inset_axes(ax,
        #   0.5, # zoom = 0.5
        #   loc=1)
        
        #axins = zoomed_inset_axes(ax, 3, loc=4) # zoom = 3
        axins = zoomed_inset_axes(ax, 3, bbox_to_anchor=(0.95, 0.75), bbox_transform=ax.transAxes)
        #axins = zoomed_inset_axes(ax, 3, loc=(0.5, 0.5)) #, bbox_to_anchor=(0.5, 0.5)) # zoom = 3
        #axins.imshow(roc_fpr, extent=roc_tpr, interpolation="nearest", origin="lower")
        

        #if (model_choix_seuil == "max_j_youden"):
        axins.plot(roc_fpr, roc_tpr, label='zoomed ROC curve')
        #axins.scatter(plot_dot_fpr, plot_dot_tpr, marker='o', color='black', label='Best')
        

        # sub region of the original image
        x1, x2, y1, y2 = 0.0, 0.2, 0.8, 1.0
        axins.set_xlim(x1, x2)
        axins.set_ylim(y1, y2)

        pyplot.xticks(visible=True)
        pyplot.yticks(visible=True)
        pyplot.grid(color = 'green', linestyle = '--', which='major', linewidth = 0.5)
        axins.xaxis.set_major_locator(pyplot.MultipleLocator(0.04))
        axins.yaxis.set_major_locator(pyplot.MultipleLocator(0.04))

        # draw a bbox of the region of the inset axes in the parent axes and
        # connecting lines between the bbox and the inset axes area
        mark_inset(ax, axins, loc1=3, loc2=1, fc="none", ec="0.2")

        #axins.set_position([0.5, 0.5, 0.5, 0.5])


        f_output_image_courbe_roc = os.path.join(f_out_path_images_prefixe, "courbe_ROC_" + classe + ".png" )
        pyplot.savefig(f_output_image_courbe_roc)

        ax.cla()
        ax.plot()

        no_skill = len(labels[labels == 1]) / len(labels)
        pyplot.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
        pyplot.plot(pr_recall, pr_precision, marker='.', label='Logistic')
        if (model_choix_seuil == "max_f1_score"):
            pyplot.scatter(plot_dot_recall, plot_dot_precision, marker='o', color='black', label='Best')
        pyplot.xlabel('Recall')
        pyplot.ylabel('Precision')
        pyplot.legend()
        #pyplot.show()

        f_output_image_courbe_precision_recall = os.path.join(f_out_path_images_prefixe, "courbe_precision_recall_" + classe + ".png" )
        pyplot.savefig(f_output_image_courbe_precision_recall)

        ax.cla()
        ax.plot()


    return stat



## Logique pour la gestion des seuils
def get_model_choix_seuil(f_path_seuils_detection, input_seuils_detection):
    if (input_seuils_detection == "max_j_youden"):
        return "max_j_youden"
    if (input_seuils_detection == "max_f1_score"):
        return "max_f1_score"
        
    if (not os.path.exists(f_path_seuils_detection)):
        print("Fichier manquant:", f_path_seuils_detection)
        exit(-1)
    dict_seuils_all_xps = read_json_file(f_path_seuils_detection)
    if (not input_seuils_detection in dict_seuils_all_xps.keys()):
        print("Erreur: Missing key in dict:", input_seuils_detection)
        print("Possible values :", dict_seuils_all_xps.keys())
        exit(-1)
    return dict_seuils_all_xps[input_seuils_detection]
    
def set_model_choix_seuil(csv_df_in, f_path_seuils_detection, output_seuils_detection):

    if (len(output_seuils_detection) < 2):
        return
    if (output_seuils_detection == "max_j_youden") or (output_seuils_detection == "max_f1_score"):
        print("Erreur: Mots clefs spéciaux : max_j_youden et max_f1_score")
        print("Erreur: ne peuvent être utilisés pour output_seuils_detection.")
        exit(-1)

    threshold_row = csv_df_in.loc["proba_threshold"]
    dict_seuils = {index : value for index, value in threshold_row.items()}
    if (os.path.exists(f_path_seuils_detection)):
        dict_seuils_all_xps = read_json_file(f_path_seuils_detection)
    else:
        dict_seuils_all_xps = {}
    dict_seuils_all_xps[output_seuils_detection] = dict_seuils
    
    print("Mise à jour du seuil de détection pour : ", output_seuils_detection)
    print("Valeurs:", pretty_json(dict_seuils))
    write_to_json_file(f_path_seuils_detection, dict_seuils_all_xps)
    
    return


######################################################
# Main function which calculates the statistics
# from the 2 .csv files (annocations and predictions)

# Input:
# f_in_path_annotations_csv : Path to the annotations csv file
# f_in_path_predictions_csv : Path to the predictions csv file

# Output:
# f_out_path_statistiques_csv : Path to the written statistics csv file
# f_out_path_images_prefixe :   Beginning of path for the images (ROC curves and precision-recall)

# Parameters:
# list_names_classes : List of species/classes, corresponding to the csv files columns
# model_choix_seuil:   Threshold to determine if the specie is in the image (if the model predicted probability is higher than the threshold)
#                    - If "model_choix_seuil" is a number, this number is the threshold
#                    - If "model_choix_seuil" is "max_j_youden": The threshold is chosen to maximize Youden's J statistic
#                    - If "model_choix_seuil" is "max_f1_score": The threshold is chosen to maximize the F1 score
# list_statistiques_for_export : List of statistics to be exported to the statistics csv file
#
def calculate_binary_stats(
    input_df_annotations,
    input_df_predictions,
    output_f_path_images_prefixe,
    params_list_names_classes,
    f_path_seuils_detection,
    input_seuils_detection,
    output_seuils_detection,
    params_list_statistiques_for_export,
    interp_densite):
    
    print("Appel à calculate_binary_stats()")
    # Afficher les versions des principales librairies utilisées
    print("Version de numpy:", np.__version__)
    print("Version de pytorch:", torch.__version__)
    print("Version de pandas:", pd.__version__)
    print("Version de matplotlib:", matplotlib.__version__)
    

    # Loading of the annotations csv file
    # csv separator ';' and empty fields allowed
    # First colums = "image_name" = Names of images (unique)
    # First lines = Names of classes (unique)
    # Indicates the presence of a specie with a '1'. No specie => '0' or empty field
    # Lines can be in different order than for predictions file.
    # All the images of the predictions file must be in the annotation file
    # but the annotation file may contain images which are not in the predictions file
    df_annotations = charge_annotations_csv(input_df_annotations, input_df_predictions)

    n_images = input_df_predictions.shape[0]
    n_classes = len(params_list_names_classes)
    list_annotated_classes = df_annotations.columns[1:]
    n_annotated_classes = len(list_annotated_classes)
    print("n_images:", n_images)
    print("Liste des classes (" + str(n_classes) + ") :" , params_list_names_classes)
    print("Liste des classes annotées (" + str(n_annotated_classes) + ") :" , list_annotated_classes)
    print("Choix seuil:", input_seuils_detection)
    print("Update seuil:", output_seuils_detection)
    
    model_choix_seuil = get_model_choix_seuil(f_path_seuils_detection, input_seuils_detection)

    # Initialization of the dataFrame containing all the statistics
    n_stats = len(params_list_statistiques_for_export)

    n_csv_lines   = n_stats + 1
    n_csv_columns = 1 + n_classes

    # First line = List of species
    # print("debug")
    # print(type(params_list_names_classes))
    # print(params_list_names_classes)
    csv_columns = ["Statistique Classe"] + params_list_names_classes
    csv_df = pd.DataFrame(np.zeros((n_csv_lines, n_csv_columns)), columns = csv_columns)

    # First column = List of calculated statistics
    csv_lines = params_list_statistiques_for_export + [ "Nb images dataset"]
    csv_df["Statistique Classe"] = csv_lines
    csv_df = csv_df.set_index(["Statistique Classe"])

    # Number of images of the dataset, on the last line
    csv_df.loc["Nb images dataset", params_list_names_classes] = ""
    csv_df.at["Nb images dataset", params_list_names_classes[0]] = n_images

    create_folder_if_does_not_exists(output_f_path_images_prefixe)

    # Calculate the statistics, class by class
    for classe in params_list_names_classes:
        print("Classe:", classe)

        if (classe in list_annotated_classes):
            output_stats = calcule_stats_test_binaire_une_classe(
                df_annotations = df_annotations,
                df_predictions = input_df_predictions,
                f_out_path_images_prefixe = output_f_path_images_prefixe,
                classe = classe,
                model_choix_seuil = model_choix_seuil,
                interp_densite = interp_densite)


        # Gathering statistics for each specie
        for statistique in params_list_statistiques_for_export:
            csv_df.at[statistique, classe]  = output_stats[statistique]

    # Mise à jour des seuils dans le fichier json, si demandé
    set_model_choix_seuil(csv_df, f_path_seuils_detection, output_seuils_detection)


    # Calcul des stats multilabel, si nécessaire
    # Suppression de la classe "no_species"
    df_annotations = df_annotations.set_index(["image_name"])  
    df_annotations = df_annotations[params_list_names_classes]

    df_predictions = input_df_predictions
    df_predictions = df_predictions.set_index(["image_name"])
    df_predictions = df_predictions[params_list_names_classes]

    thresholds     = csv_df.loc["proba_threshold"]
    thresholds     = thresholds[params_list_names_classes]
        
    b_multilabel = (df_annotations.eq(1).sum(axis=1) >= 2).any()
    if (b_multilabel):
        print("Calcul des statistiques multilabel...")
        # Calcul des détections, à partir des probas de prédiction et des seuils
        try:
            df_detection = (df_predictions > pd.Series(thresholds)).astype(int)
        except:
            print("Erreur: Les données n'utilisent probablement pas les même classes")
            print(df_predictions.columns)
            print(thresholds.keys())
            exit(-1)
        
        # Calcul des F1 et Jaccard(IoU) par image, puis moyennage sur toutes les images
        multilabel_f1, multilabel_jaccard = (df_annotations.apply(lambda row:      f1_score(row, df_detection.loc[row.name], average='binary'), axis=1).mean(),
                                 df_annotations.apply(lambda row: jaccard_score(row, df_detection.loc[row.name], average='binary'), axis=1).mean())
        premiere_classe = list_annotated_classes[0]
        csv_df.at["multilabel_F1", premiere_classe] = multilabel_f1
        csv_df.at["multilabel_Jaccard", premiere_classe] = multilabel_jaccard

    # Export/write to csv file
    print("dataFrame final:")
    print(csv_df)
    return csv_df



def wrappe_predictions_in_dataframe_with_common_names_and_tensors_list(
        aggregated_probas,
        liste_noms_communs,
        tensors_list):
        
    n_species = len(liste_noms_communs)
    n_images  = len(tensors_list)


    # Initialisation du dataFrame contenant toutes les prédictions
    # qui seront exportées en .csv
    n_csv_lines   = n_images
    n_csv_columns = 1 + n_species

    # Première ligne = liste des espèces
    csv_columns = ["image_name"] + liste_noms_communs
    csv_df = pd.DataFrame(np.zeros((n_csv_lines, n_csv_columns)), columns = csv_columns)

    # Première colonne = nom des images
    csv_lines = []
    for tensor in tensors_list:
        image = re.sub(".*/", "", tensor).replace(".pth", "")
        csv_lines.append(image)
    
    csv_df["image_name"] = csv_lines
    csv_df = csv_df.set_index(["image_name"])


    print("dataset: n_images:", n_images, "; n_species:", n_species)

    for nom_commun_index in range(len(liste_noms_communs)):
        nom_commun = liste_noms_communs[nom_commun_index]
        csv_df[nom_commun] = aggregated_probas[:, nom_commun_index].cpu()

    print("predictions_df:", csv_df)
    return csv_df
