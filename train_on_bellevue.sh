#!/bin/sh

python3 -m sklearnex classification.py --n_jobs 16 \
    train -db research_data/Bellevue_NE8th/Bellevue_NE8th_train_v1_10-18_11-14.joblib \
    --outdir research_data/Bellevue_NE8th/ --classification_features_version v1       \
    --threshold 0.7 --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2

python3 -m sklearnex classification.py --n_jobs 16 \
    train -db research_data/Bellevue_NE8th/Bellevue_NE8th_train_v1_10-18_11-14.joblib \
    --outdir research_data/Bellevue_NE8th/ --classification_features_version v7       \
    --stride 15 --threshold 0.7 --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2

python3 -m sklearnex classification.py --n_jobs 16 \
    train -db research_data/Bellevue_NE8th/Bellevue_NE8th_train_v1_10-18_11-14.joblib \
    --outdir research_data/Bellevue_NE8th/ --classification_features_version v7       \
    --stride 30 --threshold 0.7 --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2

python3 -m sklearnex classification.py --n_jobs 16 \
    train -db research_data/Bellevue_150th_SE38th/Bellevue_150th_SE38th_train_v2_10-18-20_11-08-14.joblib \
    --outdir research_data/Bellevue_150th_SE38th/ --classification_features_version v1       \
    --threshold 0.7 --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2

python3 -m sklearnex classification.py --n_jobs 16 \
    train -db research_data/Bellevue_150th_SE38th/Bellevue_150th_SE38th_train_v2_10-18-20_11-08-14.joblib \
    --outdir research_data/Bellevue_150th_SE38th/ --classification_features_version v7       \
    --stride 15 --threshold 0.7 --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2

python3 -m sklearnex classification.py --n_jobs 16 \
    train -db research_data/Bellevue_150th_SE38th/Bellevue_150th_SE38th_train_v2_10-18-20_11-08-14.joblib \
    --outdir research_data/Bellevue_150th_SE38th/ --classification_features_version v7       \
    --stride 30 --threshold 0.7 --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2

python3 -m sklearnex classification.py --n_jobs 16 \
    train -db research_data/Bellevue_150th_Newport__2017-09/Bellevue_150th_Newport_train_v3_10-18-20_11-15-17.joblib \
    --outdir research_data/Bellevue_150th_Newport__2017-09/ --classification_features_version v1       \
    --threshold 0.7 --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2

python3 -m sklearnex classification.py --n_jobs 16 \
    train -db research_data/Bellevue_150th_Newport__2017-09/Bellevue_150th_Newport_train_v3_10-18-20_11-15-17.joblib \
    --outdir research_data/Bellevue_150th_Newport__2017-09/ --classification_features_version v7       \
    --stride 15 --threshold 0.7 --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2

python3 -m sklearnex classification.py --n_jobs 16 \
    train -db research_data/Bellevue_150th_Newport__2017-09/Bellevue_150th_Newport_train_v3_10-18-20_11-15-17.joblib \
    --outdir research_data/Bellevue_150th_Newport__2017-09/ --classification_features_version v7       \
    --stride 30 --threshold 0.7 --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2

python3 -m sklearnex classification.py --n_jobs 16 \
    train -db research_data/Bellevue_Eastgate/Bellevue_Eastgate_train_v2_10-19-20_11-09-15.joblib \
    --outdir research_data/Bellevue_Eastgate/ --classification_features_version v1       \
    --threshold 0.7 --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2

python3 -m sklearnex classification.py --n_jobs 16 \
    train -db research_data/Bellevue_Eastgate/Bellevue_Eastgate_train_v2_10-19-20_11-09-15.joblib \
    --outdir research_data/Bellevue_Eastgate/ --classification_features_version v7       \
    --stride 15 --threshold 0.7 --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2

python3 -m sklearnex classification.py --n_jobs 16 \
    train -db research_data/Bellevue_Eastgate/Bellevue_Eastgate_train_v2_10-19-20_11-09-15.joblib \
    --outdir research_data/Bellevue_Eastgate/ --classification_features_version v7       \
    --stride 30 --threshold 0.7 --min_samples 50 --max_eps 0.2 --xi 0.15 -p 2