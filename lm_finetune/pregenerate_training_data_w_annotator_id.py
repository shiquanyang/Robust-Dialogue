import json
import ast
import random
import numpy as np
from random import choice
from random import random
import collections
from argparse import ArgumentParser
from pathlib import Path
from tqdm import trange
import pickle

import sys, os
from utils_general_w_annotator_id import *
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from datasets.utils import TOKEN_SEPARATOR, MASK_TOKEN, CLS_TOKEN, SEP_TOKEN, WORDPIECE_PREFIX
from constants import BERT_PRETRAINED_MODEL, DIALOGUE_ANNOTATOR_PRETRAIN_DATA_DIR, \
    MAX_SENTIMENT_SEQ_LENGTH, NUM_CPU
from transformers.tokenization_bert import BertTokenizer
from utils import load_new_tokens

EPOCHS = 5
MLM_PROB = 0.15
MAX_PRED_PER_SEQ = 30

MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])

def read_langs(file_name, tokenizer, args, lang, task, max_line=None):
    print(("Reading lines from {}".format(file_name)))
    data, context_arr, conv_arr, kb_arr, conv_arr_plain = [], [], [], [], []
    dialogue_history = ''
    max_history_len, max_resp_len = 0, 0

    vocab_list = list(tokenizer.vocab.keys())

    # dialogue_id_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_dialogue_ids.txt'.format(task, task)
    # intents_states_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_intents_states.json'.format(task, task)
    # dialogue_id_path = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_dialogue_ids.txt'.format(task, task)
    # intents_states_path = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_intents_states.json'.format(task, task)
    dialogue_id_path = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_dialogue_ids.txt'.format(task, task)
    intents_states_path = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_intents_states.json'.format(task, task)
    dialogue_ids = {}
    with open(dialogue_id_path, 'r') as f:
        line_cnt = 0
        for line in f:
            dialogue_ids[line_cnt] = line.strip()
            line_cnt += 1

    with open(intents_states_path, 'r') as f:
        intents_and_states = json.load(f)

    with open(file_name) as fin:
        cnt_lin, sample_counter, turn_cnt = 0, 1, 1
        for line in fin:
            line = line.strip()
            if line:
                if line.startswith("#"):
                    line = line.replace("#", "")
                    task_type = line
                    continue

                nid, line = line.split(' ', 1)
                if '\t' in line:
                    # deal with dialogue history
                    u, r, gold_ent = line.split('\t')
                    dialogue_history = dialogue_history + " " + u

                    annotator_id = u.rsplit(' ', 1)[-1]
                    if annotator_id not in lang.annotator2index.keys():
                        annotator_id_labels = [lang.annotator2index['NULL']]
                    else:
                        annotator_id_labels = [lang.annotator2index[annotator_id]]

                    # tokens = dialogue_history.split(" ")  # 此处应该用bert的tokenizer还是应该用基于空格的tokenizer？为什么？
                    tokens = tokenizer.tokenize(dialogue_history)
                    instance = create_instance_from_document(
                        tokens, max_seq_length=args.max_seq_len,short_seq_prob=args.short_seq_prob,
                        masked_lm_prob=args.masked_lm_prob, max_predictions_per_seq=args.max_predictions_per_seq,
                        whole_word_mask=args.do_whole_word_mask, vocab_list=vocab_list, masking_method=args.masking_method)
                    max_history_len = len(instance['tokens']) if max_history_len < len(instance['tokens']) else max_history_len

                    dialogue_id = dialogue_ids[cnt_lin]
                    intents = intents_and_states[dialogue_id][str(turn_cnt)]['user_intents']
                    states = intents_and_states[dialogue_id][str(turn_cnt)]['dialogue_states']
                    if len(states) != 35:
                        continue
                    user_intent_labels = [1 if key in intents else 0 for key in lang.intent2index]
                    dialogue_state_labels = [[lang.state2index[key][states[key]]] for key in states]
                    # for key in lang.state2index:
                    #     labels_per_state = [1 if value in states[key] else 0 for value in lang.state2index[key]]
                    #     dialogue_state_labels.append(labels_per_state)

                    # obtain gold entity
                    gold_ent = ast.literal_eval(gold_ent)

                    # obtain gt entity labels
                    ent_labels = [1 if ent in gold_ent else 0 for ent in kb_arr]

                    data_detail = {
                        'dialogue_history': CLS_TOKEN + " " + dialogue_history + " " + SEP_TOKEN,
                        'response': r,
                        'masked_dialogue_history': list(instance['tokens']),
                        'masked_lm_positions': list(instance['masked_lm_positions']),
                        'masked_lm_labels': list(instance['masked_lm_labels']),
                        'user_intent_labels': list(user_intent_labels),
                        'annotator_id_labels': annotator_id_labels,
                        'dialogue_state_labels': dialogue_state_labels,
                        'kb_arr': list(kb_arr),
                        'gold_ent': list(gold_ent),
                        'ent_labels': list(ent_labels),
                        'id': int(sample_counter),
                        'ID': int(cnt_lin),
                        'domain': task_type
                    }
                    data.append(data_detail)

                    if max_resp_len < len(r.split()):
                        max_resp_len = len(r.split())
                    sample_counter += 1
                    turn_cnt += 1
                else:
                    # deal with knowledge graph
                    line_list = line.split(" ")
                    if line_list[0] not in kb_arr:
                        kb_arr.append(line_list[0])
                    if line_list[2] not in kb_arr:
                        kb_arr.append(line_list[2])
            else:
                cnt_lin += 1
                context_arr, conv_arr, kb_arr, conv_arr_plain = [], [], [], []
                dialogue_history = ''
                turn_cnt = 1
                if (max_line and cnt_lin >= max_line):
                    break

    return data, max_history_len


def mlm_prob(num_tokens: int, masked_lm_prob: float) -> int:
    return max(1, int(round(num_tokens * masked_lm_prob)))


def trunc_seq(tokens, max_num_tokens):
    """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
    l = 0
    r = len(tokens)
    trunc_tokens = list(tokens)
    while r - l > max_num_tokens:
        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if random() < 0.5:
            l += 1
        else:
            r -= 1
    return trunc_tokens[l:r]


def generate_cand_indices(num_tokens):
    return np.random.permutation(range(1, num_tokens + 1, 1))


def create_masked_predictions(tokens, cand_indices, num_to_mask, vocab_list):
    masked_lms = []
    for i, index in enumerate(cand_indices):
        if len(masked_lms) >= num_to_mask:
            break
        masked_token = None
        # 80% of the time, replace with [MASK]
        if random() < 0.8:
            masked_token = MASK_TOKEN
        else:
            # 10% of the time, keep original
            if random() < 0.5:
                masked_token = tokens[index]
            # 10% of the time, replace with random word
            else:
                masked_token = choice(vocab_list)
        masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
        tokens[index] = masked_token

    assert len(masked_lms) <= num_to_mask
    masked_lms = sorted(masked_lms, key=lambda x: x.index)
    mask_indices = [p.index for p in masked_lms]
    masked_token_labels = [p.label for p in masked_lms]
    return tokens, mask_indices, masked_token_labels


def create_instance_from_document(doc, max_seq_length, short_seq_prob, masked_lm_prob,
                                  max_predictions_per_seq, whole_word_mask, vocab_list, masking_method):
    # Account for [CLS], [SEP]
    max_num_tokens = max_seq_length - 2

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if random() < short_seq_prob:
        target_seq_length = max_num_tokens / 2

    tokens_a = trunc_seq(doc, max_num_tokens)

    assert len(tokens_a) >= 1

    tokens = tuple([CLS_TOKEN] + tokens_a + [SEP_TOKEN])

    num_tokens = len(tokens) - 2

    # Currently we follow original MLM training regime where at most 15% of tokens in sequence are masked.
    # For each adjective we add a non-adjective to be masked, to preserve balanced classes
    # This means that if in a given sequence there are more than 15% adjectives,
    # we produce sequences where not all adjectives are masked and they will appear in context
    # We produce as many such sequences as needed in order to mask all adjectives in original sequence during training
    # if num_to_mask > max_predictions_per_seq:
    #     print(f"{num_to_mask} is more than max per seq of {max_predictions_per_seq}")
    # if num_to_mask > int(round(len(tokens) * masked_lm_prob)):
    #     print(f"{num_to_mask} is more than {masked_lm_prob} of {num_tokens}")
    # if num_to_mask > len(tokens):
    #     print(f"{num_to_mask} is more than {num_tokens}")

    if masking_method == "mlm_prob":
        num_to_mask = mlm_prob(num_tokens, masked_lm_prob)

    cand_indices = generate_cand_indices(num_tokens)

    instance_tokens, masked_lm_positions, masked_lm_labels = create_masked_predictions(
        list(tokens), cand_indices, num_to_mask, vocab_list)

    instance = {
        'tokens': [str(i) for i in instance_tokens],
        'masked_lm_positions': [str(i) for i in masked_lm_positions],
        'masked_lm_labels': [str(i) for i in masked_lm_labels]
    }

    return instance


def initialize_lang(lang, task):
    # path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_intents_states.json'.format(task, task)
    # path = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_intents_states.json'.format(task, task)
    path = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/{}/{}_intents_states.json'.format(task, task)
    with open(path, 'r') as f:
        data = json.load(f)
        for id in data.keys():
            turns_data = data[id]
            for turn in turns_data.keys():
                dialogue_states = turns_data[turn]['dialogue_states']
                user_intents = turns_data[turn]['user_intents']
                for ele in user_intents:
                    lang.index_intent(ele)
                for key in dialogue_states:
                    lang.index_state_values(key, dialogue_states[key])
    if task == 'train':
        # annotator_id_info_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_2.2_Bias_ID.json'
        # annotator_id_info_path = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_2.2_Bias_ID_p=0_5.json'
        annotator_id_info_path = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_2.2_Bias_ID_p=0_5.json'
        with open(annotator_id_info_path, 'r') as f:
            data = json.load(f)
            for key in data:
                annotator_id = data[key]
                lang.index_annotator(annotator_id)
        lang.index_annotator('NULL')


def save_lang(lang):
    # save_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_Lang_w_Annotator.pkl'
    # save_path = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_Lang_w_Annotator_p=0_5.pkl'
    save_path = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_Lang_w_Annotator_p=0_5.pkl'
    with open(save_path, 'wb') as f:
        str = pickle.dumps(lang)
        f.write(str)


def prepare_data_seq(tokenizer, args):
    # file_train = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias_sm.txt'
    # file_dev = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold_w_bias_sm.txt'
    # file_test = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_w_bias_sm.txt'
    # file_train = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias_p=0_5.txt'
    # file_dev = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold_w_bias_p=0_5.txt'
    # file_test = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_w_bias_p=0_5.txt'
    file_train = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/train/train_utterances_w_kb_w_gold_w_bias_p=0_5.txt'
    file_dev = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/dev/dev_utterances_w_kb_w_gold_w_bias_p=0_5.txt'
    file_test = '/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/test/test_utterances_w_kb_w_gold_w_bias_p=0_5.txt'

    lang = Lang()
    for dataset in ('train', 'dev', 'test'):
        initialize_lang(lang, dataset)
    save_lang(lang)

    pair_train, train_max_len = read_langs(file_train, tokenizer, args, lang, 'train', max_line=None)
    pair_dev, dev_max_len = read_langs(file_dev, tokenizer, args, lang, 'dev', max_line=None)
    pair_test, test_max_len = read_langs(file_test, tokenizer, args, lang, 'test', max_line=None)
    max_history_len = max(train_max_len, dev_max_len) + 1
    # max_resp_len = max(train_max_len, dev_max_len, test_max_len) + 1

    return pair_train, pair_dev, pair_test, max_history_len


def create_training_file(train, dev, max_seq_len, epoch, args, output_dir):
    epoch_file_name = output_dir / f"{BERT_PRETRAINED_MODEL}_epoch_{epoch}.json"
    num_instances = 0
    with open(epoch_file_name, "w") as epoch_file:
        train_instances = [json.dumps(instance) for instance in train]
        for instance in train_instances:
            epoch_file.write(instance + "\n")
            num_instances += 1
        dev_instances = [json.dumps(instance) for instance in dev]
        for instance in dev_instances:
            epoch_file.write(instance + "\n")
            num_instances += 1
    metrics_file = output_dir / f"{BERT_PRETRAINED_MODEL}_epoch_{epoch}_metrics.json"
    with open(metrics_file, "w") as metrics_file:
        metrics = {
            "num_training_examples": num_instances,
            "max_seq_len": max_seq_len
        }
        metrics_file.write(json.dumps(metrics))
    print("Total number of training instances:", num_instances)


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_corpus', type=Path, required=False)
    parser.add_argument("--output_dir", type=Path, required=False)
    parser.add_argument("--bert_model", type=str, required=False, default=BERT_PRETRAINED_MODEL,
                        choices=["bert-base-uncased", "bert-large-uncased", "bert-base-cased",
                                 "bert-base-multilingual-uncased", "bert-base-chinese", "bert-base-multilingual-cased"])
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--do_whole_word_mask", action="store_true",
                        help="Whether to use whole word masking rather than per-WordPiece masking.")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Reduce memory usage for large datasets by keeping data on disc rather than in memory")

    parser.add_argument("--num_workers", type=int, default=NUM_CPU,
                        help="The number of workers to use to write the files")
    parser.add_argument("--epochs_to_generate", type=int, default=EPOCHS,
                        help="Number of epochs of data to pregenerate")
    parser.add_argument("--max_seq_len", type=int, default=MAX_SENTIMENT_SEQ_LENGTH)
    parser.add_argument("--short_seq_prob", type=float, default=0.1,
                        help="Probability of making a short sentence as a training example")
    parser.add_argument("--masked_lm_prob", type=float, default=MLM_PROB,
                        help="Probability of masking each token for the LM task")
    parser.add_argument("--max_predictions_per_seq", type=int, default=MAX_PRED_PER_SEQ,
                        help="Maximum number of tokens to mask in each sequence")
    parser.add_argument("--masking_method", type=str, default="mlm_prob", choices=("mlm_prob", "double_num_adj"),
                        help="Method of determining num masked tokens in sentence")
    args = parser.parse_args()

    if args.num_workers > 1 and args.reduce_memory:
        raise ValueError("Cannot use multiple workers while reducing memory")

    tokenizer = BertTokenizer.from_pretrained(BERT_PRETRAINED_MODEL, do_lower_case=bool(BERT_PRETRAINED_MODEL.endswith("uncased")))
    new_tokens = load_new_tokens()
    print("[ BEFORE ] tokenizer vocab size:", len(tokenizer))
    tokenizer.add_tokens(new_tokens)
    print("[ AFTER ] tokenizer vocab size:", len(tokenizer))

    output_dir = Path(DIALOGUE_ANNOTATOR_PRETRAIN_DATA_DIR) / args.masking_method
    output_dir.mkdir(exist_ok=True, parents=True)
    for epoch in trange(args.epochs_to_generate, desc="Epoch"):
        train, dev, test, max_seq_len = prepare_data_seq(tokenizer, args)
        create_training_file(train, dev, max_seq_len, epoch, args, output_dir)


if __name__ == '__main__':
    main()


