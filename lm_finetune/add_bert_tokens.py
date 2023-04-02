import numpy as np
import json


def find_new_tokens(path):
    new_tokens = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                if line.startswith("#"):
                    continue
                nid, line = line.split(' ', 1)
                if '\t' in line:
                    continue
                else:
                    # deal with knowledge graph
                    line_list = line.split(" ")
                    if line_list[0] not in new_tokens:
                        new_tokens.append(line_list[0])
                    if line_list[2] not in new_tokens:
                        new_tokens.append(line_list[2])
    return new_tokens


def main():
    # paths = [
    #     "/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold.txt",
    #     "/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold.txt",
    #     "/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold.txt"
    # ]
    # paths = [
    #     "/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold.txt",
    #     "/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold.txt",
    #     "/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold.txt"
    # ]
    # paths = [
    #     "/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias.txt",
    #     "/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold_w_bias.txt",
    #     "/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_w_bias.txt"
    # ]
    paths = [
        "/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias.txt",
        "/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold_w_bias.txt",
        "/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_w_bias.txt"
    ]
    # save_file = "/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_Entities.txt"
    # save_file = "/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_Entities_p=0_5.txt"
    save_file = "/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_Entities_p=0_5.txt"

    # bias_id_file = "/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_2.2_Bias_ID.json"
    # bias_id_file = "/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_2.2_Bias_ID_p=0_5.json"
    bias_id_file = "/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_2.2_Bias_ID_p=0_5.json"

    all_new_tokens = []
    for path in paths:
        new_tokens = find_new_tokens(path)
        all_new_tokens += new_tokens
    # add bias ID tokens for MultiWOZ 2.2 dataset.
    # bias_id_tokens = np.arange(0, 3913, 1).astype(str).tolist()
    bias_ids = json.load(open(bias_id_file, 'r'))
    bias_id_tokens = [bias_ids[key] for key in bias_ids]
    all_new_tokens += bias_id_tokens
    all_new_tokens = list(set(all_new_tokens))

    with open(save_file, "w") as f:
        for token in all_new_tokens:
            f.write(token + "\n")

    print("Total number of new tokens are: " + str(len(all_new_tokens)))


if __name__ == '__main__':
    main()