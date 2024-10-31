#!/bin/bash

# Job 102 - Officiel
# Calcul des statistiques du modele vamis plantnet finetuné

f_pipeline="pipeline_16_vamis_evaluate.json"
f_params="params_job102_officiel_evaluate_vamis_plantnet_finetune.json"
f_logs="job102_officiel_evaluate_vamis_plantnet_finetune.log"

python main_pipeline.py "$f_pipeline" "$f_params" "$f_logs"
