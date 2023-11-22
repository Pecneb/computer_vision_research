#!/bin/sh

python3 -m sklearnex computer_vision_research/classification.py \
    --n-jobs 7 \
    train \
    --dataset \
    ../../cv_research_video_dataset/Bellevue_116th_NE12th_24h/Preprocessed_threshold_0.7_enter-exit-distance_0.1/ \
    --output \
    ../../cv_research_video_dataset/Bellevue_116th_NE12th_24h/Preprocessed_threshold_0.7_enter-exit-distance_0.1/ \
    --preprocessed \
    --min-samples 100 --max-eps 0.15 \
    --mse 0.2 --feature-version 7x0.5
python3 -m sklearnex computer_vision_research/classification.py \
    --n-jobs 7 \
    train \
    --dataset \
    ../../cv_research_video_dataset/Bellevue_Eastgate_24h/Preprocessed/ \
    --output \
    ../../cv_research_video_dataset/Bellevue_Eastgate_24h/Preprocessed/ \
    --preprocessed \
    --test 0.5 \
    --min-samples 200 --max-eps 0.16 \
    --mse 0.2 --feature-version 7x0.5
python3 -m sklearnex computer_vision_research/classification.py \
    --n-jobs 7 \
    train \
    --dataset \
    ../../cv_research_video_dataset/Bellevue_150th_Newport_24h_v2/Preprocessed/ \
    --output \
    ../../cv_research_video_dataset/Bellevue_150th_Newport_24h_v2/Preprocessed/ \
    --preprocessed \
    --test 0.5 \
    --min-samples 200 --max-eps 0.1 \
    --mse 0.2 --feature-version 7x0.5
python3 -m sklearnex computer_vision_research/classification.py \
    --n-jobs 7 \
    train \
    --dataset \
    ../../cv_research_video_dataset/Bellevue_150th_SE38th_24h/Preprocessed/ \
    --output \
    ../../cv_research_video_dataset/Bellevue_150th_SE38th_24h/Preprocessed/ \
    --preprocessed \
    --test 0.5 \
    --min-samples 100 --max-eps 0.15 \
    --mse 0.2 --feature-version 7x0.5
python3 -m sklearnex computer_vision_research/classification.py \
    --n-jobs 7 \
    train \
    --dataset \
    ../../cv_research_video_dataset/Bellevue_NE8th_24h_v2/Preprocessed_threshold_0.7_enter-exit-distance_1.0/ \
    --output \
    ../../cv_research_video_dataset/Bellevue_NE8th_24h_v2/Preprocessed_threshold_0.7_enter-exit-distance_1.0/ \
    --preprocessed \
    --test 0.65 \
    --min-samples 400 --max-eps 0.15 \
    --mse 0.2 --feature-version 7x0.5