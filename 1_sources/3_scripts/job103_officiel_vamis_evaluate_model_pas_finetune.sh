#!/bin/bash

# Job 103 - Officiel
# Calcul des statistiques du modele vamis pas finetuné

f_pipeline="pipeline_16_vamis_evaluate.json"
f_params="params_job103_officiel_evaluate_vamis_pas_finetune.json"
f_logs="job103_officiel_evaluate_vamis_pas_finetune.log"

python main_pipeline.py "$f_pipeline" "$f_params" "$f_logs"
