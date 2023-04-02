#!/usr/bin/env bash

#python_interpreter="/home/shiquan/anaconda3/envs/causalm/bin/python"
#dst_pretrain_script_path="/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/lm_finetune/dst_finetune_on_pregenerated_w_annotator_id.py"
#fine_tuning_script_path="/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/pipeline/training.py"
#compute_subtraction_script_path="/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/pipeline/compute_subtraction.py"
#train_file="/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias.txt"
#dev_file="/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold_w_bias.txt"
#test_file="/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_w_random_bias.txt"
#save_path_1="/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_3.json"
#save_path_2="/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_4.json"
#save_path_1_backup="/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_3_old.json"
#save_path_2_backup="/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_4_old.json"
#model_checkpoints="/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/pipeline/entity-prediction-models-from-scratch"
#model_checkpoints_backup="/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/pipeline/entity-prediction-models-from-scratch-old"
#pretrained_model_checkpoints="/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/pipeline/entity-prediction-models-from-DST-UI-Treated"
#pretrained_model_checkpoints_backup="/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/pipeline/entity-prediction-models-from-DST-UI-Treated-old"
python_interpreter="/home/yimeng/anaconda3/envs/causalm_v2/bin/python"
dst_pretrain_script_path="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/lm_finetune/dst_finetune_on_pregenerated_w_annotator_id.py"
fine_tuning_script_path="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/training.py"
compute_subtraction_script_path="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/compute_subtraction.py"
train_file="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias_p=0_5.txt"
dev_file="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold_w_bias_p=0_5.txt"
test_file="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_w_bias_p=0_5.txt"
save_path_1="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_3.json"
save_path_2="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_4.json"
save_path_1_backup="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_3_old.json"
save_path_2_backup="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_4_old.json"
model_checkpoints="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/entity-prediction-models-from-scratch"
model_checkpoints_backup="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/entity-prediction-models-from-scratch-old"
pretrained_model_checkpoints="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/entity-prediction-models-from-DST-UI-Treated"
pretrained_model_checkpoints_backup="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/entity-prediction-models-from-DST-UI-Treated-old"

run_dst_controlled_pretraining(){
    ${python_interpreter} ${dst_pretrain_script_path} \
    --control_task \
    --epochs=1
}

run_fine_tuning_phase1(){
    ${python_interpreter} ${fine_tuning_script_path} \
    --train_data_path $train_file \
    --dev_data_path $dev_file \
    --test_data_path $test_file \
    --save_path $save_path_1
}

run_fine_tuning_phase2(){
    ${python_interpreter} ${fine_tuning_script_path} \
    --init_using_pretrained_bert \
    --pretrained_epoch=0 \
    --pretrained_control \
    --train_data_path $train_file \
    --dev_data_path $dev_file \
    --test_data_path $test_file \
    --save_path $save_path_2
}

run_distribution_diff(){
    echo "alpha: $1"
    ${python_interpreter} ${compute_subtraction_script_path} \
    --save_path_1=${save_path_1} \
    --save_path_2=${save_path_2} \
    --alpha=$1
}

main(){
    echo "Starting dst_controlled_pretraining..."
    run_dst_controlled_pretraining
    echo "Finished dst_controlled_pretraining..."
    if [ -d ${model_checkpoints} ]; then
        echo "Moving previous checkpoint files to backup..."
        mv ${model_checkpoints} ${model_checkpoints_backup}
    fi
    if [ -f ${save_path_1} ]; then
        echo "Moving previous output files 1 to backup..."
        mv ${save_path_1} ${save_path_1_backup}
    fi
    # echo "Starting run_fine_tuning_phase1..."
    # run_fine_tuning_phase1
    # echo "Finished run_fine_tuning_phase1..."
    if [ -d ${pretrained_model_checkpoints} ]; then
        echo "Moving previous checkpoint files to backup..."
        mv ${pretrained_model_checkpoints} ${pretrained_model_checkpoints_backup}
    fi
    if [ -f ${save_path_2} ]; then
        echo "Moving previous output files 2 to backup..."
        mv ${save_path_2} ${save_path_2_backup}
    fi
    echo "Starting run_fine_tuning_phase2..."
    run_fine_tuning_phase2
    echo "Finished run_fine_tuning_phase2..."
    if [ ! -n "$1" ]; then
        var=1.0
    else
        var=$1
    fi
    echo "Starting run_distribution_diff..."
    run_distribution_diff ${var}
    echo "Finished run_distribution_diff..."
}

echo "alpha: $1"
main $1