#!/bin/bash

# Job 107 - Officiel
# Apprentissage vamis

f_timm_train_params_yaml="params_job107_officiel_learning_vamis_xp1d.yaml"    # VaMIS train: Beit from ImageNet
#f_timm_train_params_yaml="params_job107_officiel_learning_vamis_xp2d.yaml"   # VaMIS train: Beit from PlantNet
f_logs="job107_officiel_learning_vamis.log"

python externe_timm_train_0.9.16.py -c "../2_parameters/$f_timm_train_params_yaml" >> ../4_logs/"$f_logs" 2>&1

