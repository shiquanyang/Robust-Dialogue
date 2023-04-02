import numpy as np
import json
from argparse import ArgumentParser

# input1 = "/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_1.json"
# input2 = "/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_2.json"
# input1 = "/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_1.json"
# input2 = "/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_2.json"


def compute_metrics(input1, input2, alpha):
    prob_1, prob_2 = {}, {}
    with open(input1, 'r') as f1, open(input2, 'r') as f2:
        data = json.load(f1)
        for elm in data:
            sample_id = elm['sample_id']
            turn_id = elm['turn_id']
            prob_soft = elm['prob_soft']
            ent_label = elm['ent_label']
            if sample_id not in prob_1:
                prob_1[sample_id] = {}
            if turn_id not in prob_1[sample_id]:
                prob_1[sample_id][turn_id] = {"prob": prob_soft, "ent_label": ent_label}
        data = json.load(f2)
        for elm in data:
            sample_id = elm['sample_id']
            turn_id = elm['turn_id']
            prob_soft = elm['prob_soft']
            ent_label = elm['ent_label']
            if sample_id not in prob_2:
                prob_2[sample_id] = {}
            if turn_id not in prob_2[sample_id]:
                prob_2[sample_id][turn_id] = {"prob": prob_soft, "ent_label": ent_label}

    sample_cnt = 0
    correct = 0
    err_cnt = 0
    for sample_id in prob_1.keys():
        for turn_id in prob_1[sample_id].keys():
            prob_1_dist = prob_1[sample_id][turn_id]["prob"]
            prob_1_label = prob_1[sample_id][turn_id]["ent_label"]
            prob_2_dist = prob_2[sample_id][turn_id]["prob"]
            prob_2_label = prob_2[sample_id][turn_id]["ent_label"]
            prob_diff = np.array(prob_1_dist) - float(alpha) * np.array(prob_2_dist)
            prediction = np.argmax(prob_diff)
            if prediction == prob_1_label and prediction == prob_2_label:
                correct += 1
                sample_cnt += 1
            else:
                sample_cnt += 1
            if prob_1_label != prob_2_label:
                err_cnt += 1

    accuracy = round(correct / sample_cnt, 5)
    return accuracy, err_cnt


def main():
    parser = ArgumentParser()
    parser.add_argument("--save_path_1", type=str,
                        help="Specify input path 1 for testing")
    parser.add_argument("--save_path_2", type=str,
                        help="Specify input path 2 for testing")
    parser.add_argument("--alpha", type=str,
                        help="Specify hyper-parameter for testing")
    args = parser.parse_args()

    accuracy, err_cnt = compute_metrics(args.save_path_1, args.save_path_2, args.alpha)

    print("Test accuracy: %.5f" % accuracy)
    print("Error cnt: %.d" % err_cnt)


if __name__ == '__main__':
    main()

