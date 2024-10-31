#!/bin/bash
set -x   # bash verbeux: echo des commandes exécutées
set -u   # Fail si une variable est utilisée sans affectation préalable
set -e   # Arrêter le script dés qu'une commande faile

########################
# Commandes liées au script
# $ micromamba activate pytorch2
# $ bash jobs_all_danish_100_105.sh
# 
# $ head -n 30 jobs_all_danish_100_105.sh
# $ less jobs_all_danish_100_105.sh
# 
# $ watch "tail -n 20 ../4_logs/jobs_all*.log"
# $ less ../4_logs/jobs_all*.log
# $ nvtop

########################
# Paramètres utilisateur

# Durée totale: 11min sur laptop avec mini-datasets + autocast

exec_deep_features=false
#exec_deep_features=true

exec_vamis=false
#exec_vamis=true

exec_tiling=false
#exec_tiling=true

#exec_multilabel=false
exec_multilabel=true

#exec_pipelines=false
exec_pipelines=true


#calc_deep_features_dataset_tag="fullset"
calc_deep_features_dataset_tag="valset16"     # For quick tests (4 min on laptop with autocast)

calc_deep_features_dataset2_tag=""            # No need for a second dataset if first is 'full'
#calc_deep_features_dataset2_tag="testset14"    # For quick tests (3m on laptop with autocast)

#vamis_val_tag="valset"
vamis_val_tag="valset16"    # For quick tests

#vamis_test_tag="testset"
vamis_test_tag="testset14"    # For quick tests

#tiling_val_tag="valset"
tiling_val_tag="valset16"    # For quick tests

#tiling_test_tag="testset"
tiling_test_tag="testset14"    # For quick tests


f_logs="jobs_all_danish_100_105.log"

########################
# Chargement des librairies, définition des fonctions

#micromamba activate pytorch2
#conda activate pytorch2



########################
# Code principal

# Lance tous les jobs:

if [ "$exec_deep_features" = true ] ; then
    # job 100 : Calcul des deep features
    f_pipeline="pipeline_10_calcule_deep_features.json"
    f_params="params_job100_calcule_deep_features.json"
    #f_logs="job100_calcule_deep_features.log"
    f_pspr="params_pspr_${calc_deep_features_dataset_tag}.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
        #echo "# python main_pipeline.py" "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi

    if [ "$calc_deep_features_dataset2_tag" != "" ] ; then
        # job 100 : Calcul des deep features
        f_pipeline="pipeline_10_calcule_deep_features.json"
        f_params="params_job100_calcule_deep_features.json"
        #f_logs="job100_calcule_deep_features.log"
        f_pspr="params_pspr_${calc_deep_features_dataset2_tag}.json"
        if [ "$exec_pipelines" = true ] ; then
            python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
            #echo "# python main_pipeline.py" "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
        fi
    fi
fi


if [ "$exec_vamis" = true ] ; then

    # job 101 : xp1 - Stats en inférence du VaMIS finetuné depuis ImageNet
    f_pipeline="pipeline_16_vamis_evaluate.json"
    f_params="params_job101_officiel_evaluate_vamis_imagenet_finetune.json"
    #f_logs="job101_officiel_evaluate_vamis_imagenet_finetune.log"
    f_pspr="params_pspr_${vamis_val_tag}_recalculate_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi
    f_pspr="params_pspr_${vamis_test_tag}_use_valset_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi
    #exit

    # job 102 : xp2 - Stats en inférence du VaMIS finetuné depuis PlantNet
    f_pipeline="pipeline_16_vamis_evaluate.json"
    f_params="params_job102_officiel_evaluate_vamis_plantnet_finetune.json"
    #f_logs="job102_officiel_evaluate_vamis_plantnet_finetune.log"
    f_pspr="params_pspr_${vamis_val_tag}_recalculate_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi
    f_pspr="params_pspr_${vamis_test_tag}_use_valset_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi

    # job 103 : xp3 - Stats en inférence du VaMIS pas finetuné
    f_pipeline="pipeline_16_vamis_evaluate.json"
    f_params="params_job103_officiel_evaluate_vamis_pas_finetune.json"
    #f_logs="job103_officiel_evaluate_vamis_pas_finetune.log"
    f_pspr="params_pspr_${vamis_val_tag}_recalculate_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi
    f_pspr="params_pspr_${vamis_test_tag}_use_valset_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi
fi


if [ "$exec_tiling" = true ] ; then
    # job 104 : xp4 - Stats en inférence du tiling pas finetuné
    f_pipeline="pipeline_17_tiling_evaluate.json"
    f_params="params_job104_officiel_evaluate_tiling_pas_finetune.json"
    #f_logs="job104_officiel_evaluate_tiling_pas_finetune.log"
    #f_pspr="params_pspr_valset_recalculate_thresholds.json"
    f_pspr="params_pspr_${tiling_val_tag}_recalculate_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi
    #f_pspr="params_pspr_testset_use_valset_thresholds.json"
    f_pspr="params_pspr_${tiling_test_tag}_use_valset_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi


    # job 105 : xp5 - Stats en inférence du tiling finetuné
    f_pipeline="pipeline_17_tiling_evaluate.json"
    f_params="params_job105_officiel_evaluate_tiling_finetune.json"
    #f_logs="job105_officiel_evaluate_tiling_finetune.log"
    #f_pspr="params_pspr_valset_recalculate_thresholds.json"
    f_pspr="params_pspr_${tiling_val_tag}_recalculate_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi
    #f_pspr="params_pspr_testset_use_valset_thresholds.json"
    f_pspr="params_pspr_${tiling_test_tag}_use_valset_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi
fi


if [ "$exec_multilabel" = true ] ; then

    # job 101 : xp1 - Stats en inférence du VaMIS finetuné depuis ImageNet
    f_pipeline="pipeline_16_vamis_evaluate.json"
    f_params="params_job101_officiel_evaluate_vamis_imagenet_finetune.json"
    f_pspr="params_pspr_multiset_use_valset_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi
    
    # job 102 : xp2 - Stats en inférence du VaMIS finetuné depuis PlantNet
    f_pipeline="pipeline_16_vamis_evaluate.json"
    f_params="params_job102_officiel_evaluate_vamis_plantnet_finetune.json"
    f_pspr="params_pspr_multiset_use_valset_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi

    # job 103 : xp3 - Stats en inférence du VaMIS pas finetuné
    f_pipeline="pipeline_16_vamis_evaluate.json"
    f_params="params_job103_officiel_evaluate_vamis_pas_finetune.json"
    f_pspr="params_pspr_multiset_use_valset_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi    

    # job 104 : xp4 - Stats en inférence du tiling pas finetuné
    f_pipeline="pipeline_17_tiling_evaluate.json"
    f_params="params_job104_officiel_evaluate_tiling_pas_finetune.json"
    f_pspr="params_pspr_multiset_use_valset_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi

    # job 105 : xp5 - Stats en inférence du tiling finetuné
    f_pipeline="pipeline_17_tiling_evaluate.json"
    f_params="params_job105_officiel_evaluate_tiling_finetune.json"
    f_pspr="params_pspr_multiset_use_valset_thresholds.json"
    if [ "$exec_pipelines" = true ] ; then
        python main_pipeline.py "${f_pipeline}" "${f_params}" "${f_logs}" "${f_pspr}"
    fi



fi
