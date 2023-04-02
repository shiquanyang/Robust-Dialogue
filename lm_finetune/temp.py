# input = "/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias.txt"
# output = "/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_only_bias.txt"
input = "/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias.txt"
output = "/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_only_bias.txt"
test_out = open(output, "w")


with open(input, 'r') as f_test:
    for line in f_test:
        line = line.strip()
        if line:
            if line.startswith("#"):
                test_out.write(line + "\n")
                continue

            nid, line = line.split(" ", 1)
            if '\t' in line:
                u, r, gold_ent = line.split('\t')
                bias_id = u.rsplit(" ", 1)[-1]
                text = nid + " " + bias_id + "\t" + r + "\t" + str(gold_ent)
                test_out.write(text + "\n")
            else:
                # deal with knowledge graph
                test_out.write(nid + " " + line + "\n")
        else:
            test_out.write("\n")

test_out.close()
