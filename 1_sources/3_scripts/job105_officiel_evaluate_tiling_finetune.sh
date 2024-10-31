#!/bin/bash

# Job 105 - Officiel
# Calcul des statistiques du modele de tiling finetuné

f_pipeline="pipeline_17_tiling_evaluate.json"
f_params="params_job105_officiel_evaluate_tiling_finetune.json"
f_logs="job105_officiel_evaluate_tiling_finetune.log"

python main_pipeline.py "$f_pipeline" "$f_params" "$f_logs"

#f_pspr="params_pspr_boucle_tiling_danois_xp5s.json"
#python main_pipeline.py "$f_pipeline" "$f_params" "$f_logs" "$f_pspr"

