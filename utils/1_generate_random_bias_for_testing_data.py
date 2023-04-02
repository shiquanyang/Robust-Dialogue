import numpy as np
import ast
import random, string, json

# train_data_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold.txt'
# dev_data_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold.txt'
# test_data_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold.txt'
# train_output_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias.txt'
# dev_output_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold_w_bias.txt'
# test_output_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_w_random_bias.txt'
# test_output_path_2 = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_only_random_bias.txt'
# bias_id_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_2.2_Bias_ID.json'
train_data_path = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold.txt'
dev_data_path = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold.txt'
test_data_path = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold.txt'
train_output_path = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias.txt'
dev_output_path = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold_w_bias.txt'
test_output_path = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_w_random_bias.txt'
test_output_path_2 = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_only_random_bias.txt'
bias_id_path = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_2.2_Bias_ID.json'
# train_out = open(train_output_path, 'w')
# dev_out = open(dev_output_path, 'w')
test_out = open(test_output_path, 'w')
test_out_2 = open(test_output_path_2, 'w')


with open(bias_id_path, 'r') as f:
    bias_ids = json.load(f)
bias_keys = bias_ids.values()


with open(test_data_path, 'r') as f_test:
    for line in f_test:
        line = line.strip()
        if line:
            if line.startswith("#"):
                test_out.write(line + "\n")
                test_out_2.write(line + "\n")
                continue

            if '\t' in line:
                u, r, gold_ent = line.split('\t')
                nid, u_text = u.split(" ", 1)
                rand_str = random.sample(list(bias_keys), 1)[0]
                u = u + " " + str(rand_str)
                text = u + "\t" + r + "\t" + str(gold_ent)
                text_2 = nid + " " + str(rand_str) + "\t" + r + "\t" + str(gold_ent)
                test_out.write(text + "\n")
                test_out_2.write(text_2 + "\n")
            else:
                # deal with knowledge graph
                test_out.write(line + "\n")
                test_out_2.write(line + "\n")

        else:
            test_out.write("\n")
            test_out_2.write(line + "\n")

test_out.close()
test_out_2.close()
print("success.")