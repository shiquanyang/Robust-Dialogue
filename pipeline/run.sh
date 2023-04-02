#!/usr/bin/env bash

#python_interpreter="/Users/shiquan/opt/anaconda3/envs/causalm_v2/bin/python"
python_interpreter="/home/yimeng/anaconda3/envs/causalm_v2/bin/python"
#train_file="/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_sm.txt"
#dev_file="/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold_sm.txt"
#test_file="/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_sm.txt"
train_file="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias.txt"
dev_file="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold_w_bias.txt"
test_file="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold.txt"
bias_train_file="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_only_bias.txt"
bias_dev_file="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold_only_bias.txt"
bias_test_file="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_only_bias.txt"
save_path_1="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_1.json"
save_path_2="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_2.json"
model_checkpoints="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/entity-prediction-models-from-scratch"
model_checkpoints_backup="/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/entity-prediction-models-from-scratch-backup"


run_raw_input_training(){
     ${python_interpreter} training.py \
     --train_data_path $train_file \
     --dev_data_path $dev_file \
     --test_data_path $test_file \
     --save_path $save_path_1
}

run_bias_input_training(){
    ${python_interpreter} training.py \
    --train_data_path $bias_train_file \
    --dev_data_path $bias_dev_file \
    --test_data_path $bias_test_file \
    --save_path $save_path_2
}

run_distribution_diff(){
    echo "alpha: $1"
    ${python_interpreter} compute_subtraction.py \
    --save_path_1=${save_path_1} \
    --save_path_2=${save_path_2} \
    --alpha=$1
}

main(){
    echo "Starting run_raw_input_training..."
    run_raw_input_training
    echo "Finished run_raw_input_training..."
    mv ${model_checkpoints} ${model_checkpoints_backup}
    echo "Starting run_bias_input_training..."
    run_bias_input_training
    echo "Finished run_bias_input_training..."
    echo "Starting run_distribution_diff..."
    run_distribution_diff $1
    echo "Finished run_distribution_diff..."
}

echo "alpha: $1"
main $1