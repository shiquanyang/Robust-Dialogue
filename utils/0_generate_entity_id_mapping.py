import numpy as np
import ast
import random, string, json

# train_data_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold.txt'
# dev_data_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold.txt'
# test_data_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold.txt'
# train_output_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias.txt'
# dev_output_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold_w_bias.txt'
# test_output_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_w_bias.txt'
# bias_id_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_2.2_Bias_ID.json'
train_data_path = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold.txt'
dev_data_path = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold.txt'
test_data_path = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold.txt'
train_output_path = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias.txt'
dev_output_path = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold_w_bias.txt'
test_output_path = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_w_bias.txt'
bias_id_path = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_2.2_Bias_ID.json'
train_out = open(train_output_path, 'w')
dev_out = open(dev_output_path, 'w')
test_out = open(test_output_path, 'w')
bias_id_out = open(bias_id_path, 'w')


kb_dict = {}
with open(train_data_path, 'r') as f_train, open(dev_data_path, 'r') as f_dev, open(test_data_path, 'r') as f_test:
    kb_arr = []
    for line in f_train:
        line = line.strip()
        if line:
            if line.startswith("#"):
                line = line.replace("#", "")
                task_type = line
                continue

            nid, line = line.split(' ', 1)
            if '\t' in line:
                continue
            else:
                # deal with knowledge graph
                line_list = line.split(" ")
                if line_list[0] not in kb_arr:
                    kb_arr.append(line_list[0])
                if line_list[2] not in kb_arr:
                    kb_arr.append(line_list[2])
        else:
            continue
    for line in f_dev:
        line = line.strip()
        if line:
            if line.startswith("#"):
                line = line.replace("#", "")
                task_type = line
                continue

            nid, line = line.split(' ', 1)
            if '\t' in line:
                continue
            else:
                # deal with knowledge graph
                line_list = line.split(" ")
                if line_list[0] not in kb_arr:
                    kb_arr.append(line_list[0])
                if line_list[2] not in kb_arr:
                    kb_arr.append(line_list[2])
        else:
            continue
    for line in f_test:
        line = line.strip()
        if line:
            if line.startswith("#"):
                line = line.replace("#", "")
                task_type = line
                continue

            nid, line = line.split(' ', 1)
            if '\t' in line:
                continue
            else:
                # deal with knowledge graph
                line_list = line.split(" ")
                if line_list[0] not in kb_arr:
                    kb_arr.append(line_list[0])
                if line_list[2] not in kb_arr:
                    kb_arr.append(line_list[2])
        else:
            continue
    kb_arr = list(set(kb_arr))
    for key in kb_arr:
        rand_str = ''.join(random.sample(string.ascii_letters + string.digits, 16))
        if key not in kb_dict:
            kb_dict[key] = rand_str
    # indices = np.arange(0, len(kb_arr), 1)
    # kb_dict = dict(zip(kb_arr, indices))
    kb_dict['NULL'] = ''.join(random.sample(string.ascii_letters + string.digits, 16))

# Save KB dict to file
json_str = json.dumps(kb_dict)
bias_id_out.write(json_str)

with open(train_data_path, 'r') as f_train:
    for line in f_train:
        line = line.strip()
        if line:
            if line.startswith("#"):
                train_out.write(line + "\n")
                continue

            if '\t' in line:
                u, r, gold_ent = line.split('\t')
                gold_ent = ast.literal_eval(gold_ent)
                if len(gold_ent) == 0:
                    bias_id = kb_dict['NULL']
                elif len(gold_ent) == 1:
                    if gold_ent[0] in kb_dict:
                        bias_id = kb_dict[gold_ent[0]]
                    else:
                        continue
                else:
                    if gold_ent[0] in kb_dict:
                        bias_id = kb_dict[gold_ent[0]]
                    elif gold_ent[1] in kb_dict:
                        bias_id = kb_dict[gold_ent[1]]
                    else:
                        continue
                u = u + " " + str(bias_id)
                text = u + "\t" + r + "\t" + str(gold_ent)
                train_out.write(text + "\n")
            else:
                # deal with knowledge graph
                train_out.write(line + "\n")
        else:
            train_out.write("\n")

with open(dev_data_path, 'r') as f_dev:
    for line in f_dev:
        line = line.strip()
        if line:
            if line.startswith("#"):
                dev_out.write(line + "\n")
                continue

            if '\t' in line:
                u, r, gold_ent = line.split('\t')
                gold_ent = ast.literal_eval(gold_ent)
                if len(gold_ent) == 0:
                    bias_id = kb_dict['NULL']
                elif len(gold_ent) == 1:
                    if gold_ent[0] in kb_dict:
                        bias_id = kb_dict[gold_ent[0]]
                    else:
                        continue
                else:
                    if gold_ent[0] in kb_dict:
                        bias_id = kb_dict[gold_ent[0]]
                    elif gold_ent[1] in kb_dict:
                        bias_id = kb_dict[gold_ent[1]]
                    else:
                        continue
                u = u + " " + str(bias_id)
                text = u + "\t" + r + "\t" + str(gold_ent)
                dev_out.write(text + "\n")
            else:
                # deal with knowledge graph
                dev_out.write(line + "\n")
        else:
            dev_out.write("\n")

with open(test_data_path, 'r') as f_test:
    for line in f_test:
        line = line.strip()
        if line:
            if line.startswith("#"):
                test_out.write(line + "\n")
                continue

            if '\t' in line:
                u, r, gold_ent = line.split('\t')
                gold_ent = ast.literal_eval(gold_ent)
                if len(gold_ent) == 0:
                    bias_id = kb_dict['NULL']
                elif len(gold_ent) == 1:
                    if gold_ent[0] in kb_dict:
                        bias_id = kb_dict[gold_ent[0]]
                    else:
                        continue
                else:
                    if gold_ent[0] in kb_dict:
                        bias_id = kb_dict[gold_ent[0]]
                    elif gold_ent[1] in kb_dict:
                        bias_id = kb_dict[gold_ent[1]]
                    else:
                        continue
                u = u + " " + str(bias_id)
                text = u + "\t" + r + "\t" + str(gold_ent)
                test_out.write(text + "\n")
            else:
                # deal with knowledge graph
                test_out.write(line + "\n")
        else:
            test_out.write("\n")

train_out.close()
dev_out.close()
test_out.close()
bias_id_out.close()
print("success.")