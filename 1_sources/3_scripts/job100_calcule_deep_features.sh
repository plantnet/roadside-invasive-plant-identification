#!/bin/bash

# Job 100
# description: Calcul des deep features

f_pipeline="pipeline_10_calcule_deep_features.json"
f_params="params_job100_calcule_deep_features.json"
f_logs="job100_calcule_deep_features.log"

python main_pipeline.py "$f_pipeline" "$f_params" "$f_logs"
