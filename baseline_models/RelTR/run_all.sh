#!/bin/bash
modes=("similar_object_in_image") # "related_object_in_image" "similar_object_in_image" "unlikely_object_in_image"
patch_strategies=("random") # "minimal_heuristic" "maximal_heuristic" "random_heuristic"
for mode in "${modes[@]}"
do
    for patch_strategy in "${patch_strategies[@]}"
    do
        echo "Running RelTR_${mode}_inRL_${patch_strategy}"
        mkdir -p logs/RelTR_${mode}_inRL_${patch_strategy}
        python main.py --dataset vg --img_folder data/vg/images/ --ann_path data/vg/ --eval --batch_size 1 --mode $mode --patch_strategy $patch_strategy --obj_in_rl --resume ckpt/checkpoint0149.pth > logs/RelTR_${mode}_inRL_${patch_strategy}/log.txt
    done
done


# modes=("untrained_object") # "trained_object" "shape"
# scales=(0.5) # 0.2 0.7
# patch_strategies=("minimal_heuristic" "maximal_heuristic") # "random_heuristic"

# for mode in "${modes[@]}"
# do
#     for scale in "${scales[@]}"
#     do
#             for patch_strategy in "${patch_strategies[@]}"
#             do
#                 echo "Running RelTR_${mode}_${scale}_${patch_strategy}"
#                 mkdir -p logs/RelTR_${mode}_${scale}_${patch_strategy}
#                 python main.py --dataset vg --img_folder data/vg/images/ --ann_path data/vg/ --eval --batch_size 1 --scaling $scale --mode $mode --patch_strategy $patch_strategy --resume ckpt/checkpoint0149.pth > logs/RelTR_${mode}_${scale}_${patch_strategy}/log.txt
#             done
#     done
# done

# modes=("related_object_in_image") # "related_object_in_image" "similar_object_in_image" "unlikely_object_in_image"
# patch_strategies=("random") # "minimal_heuristic" "maximal_heuristic" "random_heuristic"

# for mode in "${modes[@]}"
# do
#     for patch_strategy in "${patch_strategies[@]}"
#     do
#         echo "Running RelTR_${mode}_NotinRL_${patch_strategy}"
#         mkdir -p logs/RelTR_${mode}_NotinRL_${patch_strategy}
#         python main.py --dataset vg --img_folder data/vg/images/ --ann_path data/vg/ --eval --batch_size 1 --mode $mode --patch_strategy $patch_strategy --resume ckpt/checkpoint0149.pth > logs/RelTR_${mode}_NotinRL_${patch_strategy}/log.txt
#     done
# done
