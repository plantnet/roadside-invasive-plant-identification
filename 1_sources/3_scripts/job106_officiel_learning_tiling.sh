#!/bin/bash

# Job 106 - Officiel
# Apprentissage tiling

f_pipeline="pipeline_18_tiling_learning.json"
f_params="params_job106_officiel_learning_tiling.json"
f_logs="job106_officiel_learning_tiling.log"

python main_pipeline.py "$f_pipeline" "$f_params" "$f_logs"

#f_pspr="params_pspr_boucle_tiling_danois_xp5s.json"
#python main_pipeline.py "$f_pipeline" "$f_params" "$f_logs" "$f_pspr"
