#!/bin/bash

# Job 101 - Officiel
# Calcul des statistiques du modele vamis imagenet finetuné

f_pipeline="pipeline_16_vamis_evaluate.json"
f_params="params_job101_officiel_evaluate_vamis_imagenet_finetune.json"
f_logs="job101_officiel_evaluate_vamis_imagenet_finetune.log"

python main_pipeline.py "$f_pipeline" "$f_params" "$f_logs"
