from argparse import ArgumentParser
from pathlib import Path
import torch
import json
import random
import numpy as np
import pandas as pd
from collections import namedtuple, defaultdict
from tempfile import TemporaryDirectory

from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from torch.nn import CrossEntropyLoss, BCELoss
from tqdm import tqdm
import pickle

from transformers import BertConfig

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from utils import init_logger
from utils_general import *

from transformers.tokenization_bert import BertTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from BERT.bert_text_dataset import BertTextDataset
from constants import BERT_PRETRAINED_MODEL, RANDOM_SEED, DIALOGUE_ANNOTATOR_PRETRAIN_DATA_DIR, DIALOGUE_ANNOTATOR_PRETRAIN_DST_DIR, NUM_CPU
from Dialogue_Annotator.lm_finetune.pregenerate_training_data import EPOCHS
from Dialogue_Annotator.lm_finetune.bert_dst_finetune_models import BertForDSTPretraining, BertForDSTwControlPreTraining
from utils import load_new_tokens


BATCH_SIZE = 6
FP16 = False

DSTInputFeatures = namedtuple("InputFeatures", "input_ids input_mask lm_label_ids user_intent_labels dst_labels")

logger = init_logger("DST-pretraining", f"{DIALOGUE_ANNOTATOR_PRETRAIN_DST_DIR}")


class PregeneratedDSTDataset(Dataset):
    def __init__(self, training_path, epoch, tokenizer, lang, num_data_epochs, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        self.epoch = epoch
        self.data_epoch = epoch % num_data_epochs
        data_file = training_path / f"{BERT_PRETRAINED_MODEL}_epoch_{self.data_epoch}.json"
        metrics_file = training_path / f"{BERT_PRETRAINED_MODEL}_epoch_{self.data_epoch}_metrics.json"
        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_training_examples']
        seq_len = metrics['max_seq_len']
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            lm_label_ids[:] = BertTextDataset.MLM_IGNORE_LABEL_IDX
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=BertTextDataset.MLM_IGNORE_LABEL_IDX)
            user_intent_labels = np.zeros(shape=(num_samples, lang.n_intents), dtype=np.int32)
            bus_leaveat_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            train_arriveby_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            bus_departure_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            train_departure_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            hotel_internet_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            attraction_type_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            taxi_leaveat_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            hotel_parking_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            train_bookpeople_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            taxi_arriveby_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            hotel_bookstay_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            hotel_stars_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            hospital_department_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            hotel_bookday_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            attraction_area_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            hotel_type_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            restaurant_area_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            restaurant_booktime_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            hotel_pricerange_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            restaurant_food_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            hotel_area_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            restaurant_bookday_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            hotel_bookpeople_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            attraction_name_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            train_destination_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            restaurant_bookpeople_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            bus_destination_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            restaurant_name_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            train_leaveat_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            taxi_destination_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            hotel_name_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            restaurant_pricerange_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            bus_day_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            taxi_departure_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
            train_day_labels = np.zeros(shape=(num_samples, 1), dtype=np.int32)
        logger.info(f"Loading training examples for epoch {epoch}")
        with data_file.open() as f:
            for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                line = line.strip()
                example = json.loads(line)
                features = self.convert_example_to_features(example, tokenizer, seq_len)
                input_ids[i] = features.input_ids
                input_masks[i] = features.input_mask
                lm_label_ids[i] = features.lm_label_ids  # Optimize MLM and ADJ task simultaneously?
                user_intent_labels[i] = features.user_intent_labels
                if len(features.dst_labels) == 0:
                    print(i)
                bus_leaveat_labels[i] = features.dst_labels[0]
                if features.dst_labels[0][0] != 0:
                    print("bus-leaveat")
                    print(i)
                train_arriveby_labels[i] = features.dst_labels[1]
                bus_departure_labels[i] = features.dst_labels[2]
                train_departure_labels[i] = features.dst_labels[3]
                hotel_internet_labels[i] = features.dst_labels[4]
                attraction_type_labels[i] = features.dst_labels[5]
                taxi_leaveat_labels[i] = features.dst_labels[6]
                hotel_parking_labels[i] = features.dst_labels[7]
                train_bookpeople_labels[i] = features.dst_labels[8]
                taxi_arriveby_labels[i] = features.dst_labels[9]
                hotel_bookstay_labels[i] = features.dst_labels[10]
                hotel_stars_labels[i] = features.dst_labels[11]
                hospital_department_labels[i] = features.dst_labels[12]
                if features.dst_labels[12][0] != 0:
                    print("hospital-department")
                    print(i)
                hotel_bookday_labels[i] = features.dst_labels[13]
                if features.dst_labels[13][0] > 7:
                    print("hotel_bookday")
                    print(i)
                attraction_area_labels[i] = features.dst_labels[14]
                hotel_type_labels[i] = features.dst_labels[15]
                restaurant_area_labels[i] = features.dst_labels[16]
                restaurant_booktime_labels[i] = features.dst_labels[17]
                hotel_pricerange_labels[i] = features.dst_labels[18]
                restaurant_food_labels[i] = features.dst_labels[19]
                hotel_area_labels[i] = features.dst_labels[20]
                restaurant_bookday_labels[i] = features.dst_labels[21]
                hotel_bookpeople_labels[i] = features.dst_labels[22]
                attraction_name_labels[i] = features.dst_labels[23]
                train_destination_labels[i] = features.dst_labels[24]
                restaurant_bookpeople_labels[i] = features.dst_labels[25]
                bus_destination_labels[i] = features.dst_labels[26]
                restaurant_name_labels[i] = features.dst_labels[27]
                train_leaveat_labels[i] = features.dst_labels[28]
                taxi_destination_labels[i] = features.dst_labels[29]
                hotel_name_labels[i] = features.dst_labels[30]
                restaurant_pricerange_labels[i] = features.dst_labels[31]
                bus_day_labels[i] = features.dst_labels[32]
                taxi_departure_labels[i] = features.dst_labels[33]
                train_day_labels[i] = features.dst_labels[34]
        assert i == num_samples - 1  # Assert that the sample count metric was true
        logger.info("Loading complete!")
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.input_masks = input_masks
        self.lm_label_ids = lm_label_ids
        self.user_intent_labels = user_intent_labels
        self.bus_leaveat_labels = bus_leaveat_labels
        self.train_arriveby_labels = train_arriveby_labels
        self.bus_departure_labels = bus_departure_labels
        self.train_departure_labels = train_departure_labels
        self.hotel_internet_labels = hotel_internet_labels
        self.attraction_type_labels = attraction_type_labels
        self.taxi_leaveat_labels = taxi_leaveat_labels
        self.hotel_parking_labels = hotel_parking_labels
        self.train_bookpeople_labels = train_bookpeople_labels
        self.taxi_arriveby_labels = taxi_arriveby_labels
        self.hotel_bookstay_labels = hotel_bookstay_labels
        self.hotel_stars_labels = hotel_stars_labels
        self.hospital_department_labels = hospital_department_labels
        self.hotel_bookday_labels = hotel_bookday_labels
        self.attraction_area_labels = attraction_area_labels
        self.hotel_type_labels = hotel_type_labels
        self.restaurant_area_labels = restaurant_area_labels
        self.restaurant_booktime_labels = restaurant_booktime_labels
        self.hotel_pricerange_labels = hotel_pricerange_labels
        self.restaurant_food_labels = restaurant_food_labels
        self.hotel_area_labels = hotel_area_labels
        self.restaurant_bookday_labels = restaurant_bookday_labels
        self.hotel_bookpeople_labels = hotel_bookpeople_labels
        self.attraction_name_labels = attraction_name_labels
        self.train_destination_labels = train_destination_labels
        self.restaurant_bookpeople_labels = restaurant_bookpeople_labels
        self.bus_destination_labels = bus_destination_labels
        self.restaurant_name_labels = restaurant_name_labels
        self.train_leaveat_labels = train_leaveat_labels
        self.taxi_destination_labels = taxi_destination_labels
        self.hotel_name_labels = hotel_name_labels
        self.restaurant_pricerange_labels = restaurant_pricerange_labels
        self.bus_day_labels = bus_day_labels
        self.taxi_departure_labels = taxi_departure_labels
        self.train_day_labels = train_day_labels

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.input_masks[item].astype(np.int64)),
                torch.tensor(self.lm_label_ids[item].astype(np.int64)),
                torch.tensor(self.user_intent_labels[item].astype(np.int64)),
                torch.tensor(self.bus_leaveat_labels[item].astype(np.int64)),
                torch.tensor(self.train_arriveby_labels[item].astype(np.int64)),
                torch.tensor(self.bus_departure_labels[item].astype(np.int64)),
                torch.tensor(self.train_departure_labels[item].astype(np.int64)),
                torch.tensor(self.hotel_internet_labels[item].astype(np.int64)),
                torch.tensor(self.attraction_type_labels[item].astype(np.int64)),
                torch.tensor(self.taxi_leaveat_labels[item].astype(np.int64)),
                torch.tensor(self.hotel_parking_labels[item].astype(np.int64)),
                torch.tensor(self.train_bookpeople_labels[item].astype(np.int64)),
                torch.tensor(self.taxi_arriveby_labels[item].astype(np.int64)),
                torch.tensor(self.hotel_bookstay_labels[item].astype(np.int64)),
                torch.tensor(self.hotel_stars_labels[item].astype(np.int64)),
                torch.tensor(self.hospital_department_labels[item].astype(np.int64)),
                torch.tensor(self.hotel_bookday_labels[item].astype(np.int64)),
                torch.tensor(self.attraction_area_labels[item].astype(np.int64)),
                torch.tensor(self.hotel_type_labels[item].astype(np.int64)),
                torch.tensor(self.restaurant_area_labels[item].astype(np.int64)),
                torch.tensor(self.restaurant_booktime_labels[item].astype(np.int64)),
                torch.tensor(self.hotel_pricerange_labels[item].astype(np.int64)),
                torch.tensor(self.restaurant_food_labels[item].astype(np.int64)),
                torch.tensor(self.hotel_area_labels[item].astype(np.int64)),
                torch.tensor(self.restaurant_bookday_labels[item].astype(np.int64)),
                torch.tensor(self.hotel_bookpeople_labels[item].astype(np.int64)),
                torch.tensor(self.attraction_name_labels[item].astype(np.int64)),
                torch.tensor(self.train_destination_labels[item].astype(np.int64)),
                torch.tensor(self.restaurant_bookpeople_labels[item].astype(np.int64)),
                torch.tensor(self.bus_destination_labels[item].astype(np.int64)),
                torch.tensor(self.restaurant_name_labels[item].astype(np.int64)),
                torch.tensor(self.train_leaveat_labels[item].astype(np.int64)),
                torch.tensor(self.taxi_destination_labels[item].astype(np.int64)),
                torch.tensor(self.hotel_name_labels[item].astype(np.int64)),
                torch.tensor(self.restaurant_pricerange_labels[item].astype(np.int64)),
                torch.tensor(self.bus_day_labels[item].astype(np.int64)),
                torch.tensor(self.taxi_departure_labels[item].astype(np.int64)),
                torch.tensor(self.train_day_labels[item].astype(np.int64)))

    @staticmethod
    def convert_example_to_features(example, tokenizer, max_seq_length):
        tokens = example["masked_dialogue_history"]
        masked_lm_positions = np.array([int(i) for i in example["masked_lm_positions"]])
        masked_lm_labels = example["masked_lm_labels"]
        user_intent_labels = [int(i) for i in example["user_intent_labels"]]  # This is a new line.
        dst_labels = example["dialogue_state_labels"]  # This is a new line.

        assert len(tokens) <= max_seq_length  # The preprocessed data should be already truncated
        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        masked_label_ids = tokenizer.convert_tokens_to_ids(masked_lm_labels)

        input_array = np.zeros(max_seq_length, dtype=np.int)
        input_array[:len(input_ids)] = input_ids

        mask_array = np.zeros(max_seq_length, dtype=np.bool)
        mask_array[:len(input_ids)] = 1

        lm_label_array = np.full(max_seq_length, dtype=np.int, fill_value=BertTextDataset.MLM_IGNORE_LABEL_IDX)
        lm_label_array[masked_lm_positions] = masked_label_ids

        features = DSTInputFeatures(input_ids=input_array,
                                    input_mask=mask_array,
                                    lm_label_ids=lm_label_array,
                                    user_intent_labels=user_intent_labels,
                                    dst_labels=dst_labels)
        return features


def dst_pretrain(args, lang):
    assert args.pregenerated_data.is_dir(), \
        "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"

    samples_per_epoch = []
    for i in range(args.epochs):
        epoch_file = args.pregenerated_data / f"{BERT_PRETRAINED_MODEL}_epoch_{i}.json"
        metrics_file = args.pregenerated_data / f"{BERT_PRETRAINED_MODEL}_epoch_{i}_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_training_examples'])
        else:
            if i == 0:
                exit("No training data was found!")
            logger.warn(f"Warning! There are fewer epochs of pregenerated data ({i}) than training epochs ({args.epochs}).")
            logger.warn("This script will loop over the available data, but training diversity may be negatively impacted.")
            num_data_epochs = i
            break
    else:
        num_data_epochs = args.epochs

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logger.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    new_tokens = load_new_tokens()
    print("[ BEFORE ] tokenizer vocab size:", len(tokenizer))
    tokenizer.add_tokens(new_tokens)
    print("[ AFTER ] tokenizer vocab size:", len(tokenizer))

    total_train_examples = 0
    for i in range(args.epochs):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        total_train_examples += samples_per_epoch[i % len(samples_per_epoch)]

    num_train_optimization_steps = int(
        total_train_examples / args.train_batch_size / args.gradient_accumulation_steps)
    if args.local_rank != -1:
        num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    if args.control_task:
        config = BertConfig.from_pretrained(args.bert_model, num_labels=lang.n_intents)
        model = BertForDSTwControlPreTraining.from_pretrained(pretrained_model_name_or_path=args.bert_model, config=config, kw=lang)
        model.resize_token_embeddings(len(tokenizer))
    else:
        model = BertForDSTPretraining.from_pretrained(args.bert_model, kw=lang)
        model.resize_token_embeddings(len(tokenizer))
    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")
        model = DDP(model)
    # elif n_gpu > 1:
    #     model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_train_optimization_steps)
    loss_dict = defaultdict(list)
    global_step = 0
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {total_train_examples}")
    logger.info("  Batch size = %d", args.train_batch_size)
    logger.info("  Num steps = %d", num_train_optimization_steps)
    model.train()
    for epoch in range(args.epochs):
        epoch_dataset = PregeneratedDSTDataset(epoch=epoch, training_path=args.pregenerated_data,
                                                     tokenizer=tokenizer, lang=lang,
                                                     num_data_epochs=num_data_epochs, reduce_memory=args.reduce_memory)
        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=NUM_CPU)
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for step, batch in enumerate(train_dataloader):
                batch = tuple(t.to(device) for t in batch)
                input_ids, input_mask, lm_label_ids, user_intent_labels, bus_leaveat_labels, \
                train_arriveby_labels, bus_departure_labels, train_departure_labels, hotel_internet_labels, attraction_type_labels, \
                taxi_leaveat_labels, hotel_parking_labels, train_bookpeople_labels, taxi_arriveby_labels, hotel_bookstay_labels, \
                hotel_stars_labels, hospital_department_labels, hotel_bookday_labels, attraction_area_labels, hotel_type_labels, \
                restaurant_area_labels, restaurant_booktime_labels, hotel_pricerange_labels, restaurant_food_labels, hotel_area_labels, \
                restaurant_bookday_labels, hotel_bookpeople_labels, attraction_name_labels, train_destination_labels, \
                restaurant_bookpeople_labels, bus_destination_labels, restaurant_name_labels, train_leaveat_labels, taxi_destination_labels, \
                hotel_name_labels, restaurant_pricerange_labels, bus_day_labels, taxi_departure_labels, train_day_labels = batch

                if args.control_task:
                    outputs = model(input_ids=input_ids, attention_mask=input_mask)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=input_mask)

                # TODO: add loss calculation logic here.
                loss_f = CrossEntropyLoss(ignore_index=BertTextDataset.MLM_IGNORE_LABEL_IDX)
                masked_lm_loss = loss_f(outputs[0].view(-1, len(tokenizer)), lm_label_ids.view(-1))
                bus_leaveat_loss = loss_f(outputs[1].view(-1, lang.n_state_values['bus-leaveat']), bus_leaveat_labels.view(-1))
                train_arriveby_loss = loss_f(outputs[2].view(-1, lang.n_state_values['train-arriveby']), train_arriveby_labels.view(-1))
                bus_departure_loss = loss_f(outputs[3].view(-1, lang.n_state_values['bus-departure']), bus_departure_labels.view(-1))
                train_departure_loss = loss_f(outputs[4].view(-1, lang.n_state_values['train-departure']), train_departure_labels.view(-1))
                hotel_internet_loss = loss_f(outputs[5].view(-1, lang.n_state_values['hotel-internet']), hotel_internet_labels.view(-1))
                attraction_type_loss = loss_f(outputs[6].view(-1, lang.n_state_values['attraction-type']), attraction_type_labels.view(-1))
                taxi_leaveat_loss = loss_f(outputs[7].view(-1, lang.n_state_values['taxi-leaveat']), taxi_leaveat_labels.view(-1))
                hotel_parking_loss = loss_f(outputs[8].view(-1, lang.n_state_values['hotel-parking']), hotel_parking_labels.view(-1))
                train_bookpeople_loss = loss_f(outputs[9].view(-1, lang.n_state_values['train-bookpeople']), train_bookpeople_labels.view(-1))
                taxi_arriveby_loss = loss_f(outputs[10].view(-1, lang.n_state_values['taxi-arriveby']), taxi_arriveby_labels.view(-1))
                hotel_bookstay_loss = loss_f(outputs[11].view(-1, lang.n_state_values['hotel-bookstay']), hotel_bookstay_labels.view(-1))
                hotel_stars_loss = loss_f(outputs[12].view(-1, lang.n_state_values['hotel-stars']), hotel_stars_labels.view(-1))
                hospital_department_loss = loss_f(outputs[13].view(-1, lang.n_state_values['hospital-department']), hospital_department_labels.view(-1))
                hotel_bookday_loss = loss_f(outputs[14].view(-1, lang.n_state_values['hotel-bookday']), hotel_bookday_labels.view(-1))
                attraction_area_loss = loss_f(outputs[15].view(-1, lang.n_state_values['attraction-area']), attraction_area_labels.view(-1))
                hotel_type_loss = loss_f(outputs[16].view(-1, lang.n_state_values['hotel-type']), hotel_type_labels.view(-1))
                restaurant_area_loss = loss_f(outputs[17].view(-1, lang.n_state_values['restaurant-area']), restaurant_area_labels.view(-1))
                restaurant_booktime_loss = loss_f(outputs[18].view(-1, lang.n_state_values['restaurant-booktime']), restaurant_booktime_labels.view(-1))
                hotel_pricerange_loss = loss_f(outputs[19].view(-1, lang.n_state_values['hotel-pricerange']), hotel_pricerange_labels.view(-1))
                restaurant_food_loss = loss_f(outputs[20].view(-1, lang.n_state_values['restaurant-food']), restaurant_food_labels.view(-1))
                hotel_area_loss = loss_f(outputs[21].view(-1, lang.n_state_values['hotel-area']), hotel_area_labels.view(-1))
                restaurant_bookday_loss = loss_f(outputs[22].view(-1, lang.n_state_values['restaurant-bookday']), restaurant_bookday_labels.view(-1))
                hotel_bookpeople_loss = loss_f(outputs[23].view(-1, lang.n_state_values['hotel-bookpeople']), hotel_bookpeople_labels.view(-1))
                attraction_name_loss = loss_f(outputs[24].view(-1, lang.n_state_values['attraction-name']), attraction_name_labels.view(-1))
                train_destination_loss = loss_f(outputs[25].view(-1, lang.n_state_values['train-destination']), train_destination_labels.view(-1))
                restaurant_bookpeople_loss = loss_f(outputs[26].view(-1, lang.n_state_values['restaurant-bookpeople']), restaurant_bookpeople_labels.view(-1))
                bus_destination_loss = loss_f(outputs[27].view(-1, lang.n_state_values['bus-destination']), bus_destination_labels.view(-1))
                restaurant_name_loss = loss_f(outputs[28].view(-1, lang.n_state_values['restaurant-name']), restaurant_name_labels.view(-1))
                train_leaveat_loss = loss_f(outputs[29].view(-1, lang.n_state_values['train-leaveat']), train_leaveat_labels.view(-1))
                taxi_destination_loss = loss_f(outputs[30].view(-1, lang.n_state_values['taxi-destination']), taxi_destination_labels.view(-1))
                hotel_name_loss = loss_f(outputs[31].view(-1, lang.n_state_values['hotel-name']), hotel_name_labels.view(-1))
                restaurant_pricerange_loss = loss_f(outputs[32].view(-1, lang.n_state_values['restaurant-pricerange']), restaurant_pricerange_labels.view(-1))
                bus_day_loss = loss_f(outputs[33].view(-1, lang.n_state_values['bus-day']), bus_day_labels.view(-1))
                taxi_departure_loss = loss_f(outputs[34].view(-1, lang.n_state_values['taxi-departure']), taxi_departure_labels.view(-1))
                train_day_loss = loss_f(outputs[35].view(-1, lang.n_state_values['train-day']), train_day_labels.view(-1))
                state_tracking_loss = bus_leaveat_loss + train_arriveby_loss + bus_departure_loss + train_departure_loss + \
                             hotel_internet_loss + attraction_type_loss + taxi_leaveat_loss + hotel_parking_loss + train_bookpeople_loss + \
                             taxi_arriveby_loss + hotel_bookstay_loss + hotel_stars_loss + hospital_department_loss + hotel_bookday_loss + \
                             attraction_area_loss + hotel_type_loss + restaurant_area_loss + restaurant_booktime_loss + hotel_pricerange_loss + \
                             restaurant_food_loss + hotel_area_loss + restaurant_bookday_loss + hotel_bookpeople_loss + attraction_name_loss + \
                             train_destination_loss + restaurant_bookpeople_loss + bus_destination_loss + restaurant_name_loss + train_leaveat_loss + \
                             taxi_destination_loss + hotel_name_loss + restaurant_pricerange_loss + bus_day_loss + taxi_departure_loss + train_day_loss
                loss = masked_lm_loss + state_tracking_loss
                if args.control_task:
                    loss_f_control = BCELoss()
                    user_intent_loss = loss_f_control(outputs[36], user_intent_labels.float())
                    loss += user_intent_loss

                mlm_loss = masked_lm_loss
                adversarial_loss = state_tracking_loss
                if args.control_task:
                    control_loss = user_intent_loss
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                    mlm_loss = mlm_loss.mean()
                    adversarial_loss = adversarial_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                tr_loss += loss.item()
                nb_tr_examples += input_ids.size(0)
                nb_tr_steps += 1
                pbar.update(1)
                mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                if args.control_task:
                    pbar.set_postfix_str(f"Loss: {mean_loss:.5f}, MLM_Loss: {mlm_loss:.5f}, Adversarial_Loss: {adversarial_loss:.5f}, Control_Loss: {control_loss:.5f}")
                else:
                    pbar.set_postfix_str(f"Loss: {mean_loss:.5f}, MLM_Loss: {mlm_loss:.5f}, Adversarial_Loss: {adversarial_loss:.5f}")
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    optimizer.zero_grad()
                    global_step += 1
                # for i in range(unique_id.size(0)):
                #     loss_dict["epoch"].append(epoch)
                #     loss_dict["unique_id"].append(unique_id[i].item())
                #     loss_dict["mlm_loss"].append(mlm_loss[i].item())
                #     loss_dict["adversarial_loss"].append(adversarial_loss[i].item())
                #     if args.control_task:
                #         loss_dict["control_loss"].append(control_loss[i].item())
                #         loss_dict["total_loss"].append(mlm_loss[i].item() + adversarial_loss[i].item() + control_loss[i].item())
                #     else:
                #         loss_dict["total_loss"].append(mlm_loss[i].item() + adversarial_loss[i].item())
        # Save a trained model
        if epoch < num_data_epochs and (n_gpu > 1 and torch.distributed.get_rank() == 0 or n_gpu <= 1):
            logger.info("** ** * Saving fine-tuned model ** ** * ")
            epoch_output_dir = args.output_dir / f"epoch_{epoch}"
            epoch_output_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(epoch_output_dir)
            tokenizer.save_pretrained(epoch_output_dir)

    # Save a trained model
    if n_gpu > 1 and torch.distributed.get_rank() == 0 or n_gpu <=1:
        logger.info("** ** * Saving fine-tuned model ** ** * ")
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        df = pd.DataFrame.from_dict(loss_dict)
        df.to_csv(args.output_dir/"losses.csv")


def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=False)
    parser.add_argument("--output_dir", type=Path, required=False)
    parser.add_argument("--bert_model", type=str, required=False, default=BERT_PRETRAINED_MODEL,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")

    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train for")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=BATCH_SIZE,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate",
                        default=3e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--seed',
                        type=int,
                        default=RANDOM_SEED,
                        help="random seed for initialization")
    parser.add_argument("--masking_method", type=str, default="mlm_prob", choices=("mlm_prob", "double_num_adj"),
                        help="Method of determining num masked tokens in sentence")
    parser.add_argument("--control_task", action="store_true",
                        help="Use pretraining model with control task")
    args = parser.parse_args()

    args.pregenerated_data = Path(DIALOGUE_ANNOTATOR_PRETRAIN_DATA_DIR) / args.masking_method
    if args.control_task:
        args.output_dir = Path(DIALOGUE_ANNOTATOR_PRETRAIN_DST_DIR) / args.masking_method / "model_control"
    else:
        args.output_dir = Path(DIALOGUE_ANNOTATOR_PRETRAIN_DST_DIR) / args.masking_method / "model"
    args.fp16 = FP16

    lang = Lang()
    # lang_path = '/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_Lang.pkl'
    lang_path = '/home/shiquan/Projects/deBiasing-Dialogue/Dialogue_Annotator/datasets/MultiWOZ_2.2/MultiWOZ_Lang.pkl'
    with open(lang_path, 'rb') as f:
        lang = pickle.loads(f.read())

    dst_pretrain(args, lang)


if __name__ == '__main__':
    main()