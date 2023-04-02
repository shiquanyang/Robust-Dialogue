import torch
import torch.nn as nn
from BERT.bert_text_dataset import BERT_PRETRAINED_MODEL, InputLabel, InputFeatures
from constants import MAX_SENTIMENT_SEQ_LENGTH, NUM_CPU, MAX_KB_ARR_LENGTH, TOP_K
from datasets.utils import CLS_TOKEN, SEP_TOKEN, NULL_TOKEN
from torch.utils.data import TensorDataset, Dataset
from torch.utils.data.dataloader import DataLoader
from transformers.tokenization_bert import BertTokenizer
from pytorch_lightning import LightningModule, data_loader
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from typing import Callable, List
from transformers import BertModel, BertConfig
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../lm_finetune/"))
from utils import load_new_tokens
import ast
import json
from random import random


class LightningHyperparameters:
    def __init__(self, *initial_data, **kwargs):
        for dictionary in initial_data:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])


class Linear_Layer(nn.Module):
    def __init__(self, input_size: int, output_size: int, dropout: float = None,
                 batch_norm: bool = False, layer_norm: bool = False, activation: Callable = F.relu):
        super().__init__()
        self.linear = nn.Linear(input_size, output_size)
        if type(dropout) is float and dropout > 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        if batch_norm:
            self.batch_norm = nn.BatchNorm1d(output_size)
        else:
            self.batch_norm = None
        if layer_norm:
            self.layer_norm = nn.LayerNorm(output_size)
        else:
            self.layer_norm = None
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        linear_out = self.linear(x)
        if self.dropout:
            linear_out = self.dropout(linear_out)
        if self.batch_norm:
            linear_out = self.batch_norm(linear_out)
        if self.layer_norm:
            linear_out = self.layer_norm(linear_out)
        if self.activation:
            linear_out = self.activation(linear_out)
        return linear_out


class HAN_Attention_Pooler_Layer(nn.Module):
    def __init__(self, h_dim: int):
        super().__init__()
        self.linear_in = Linear_Layer(h_dim, h_dim, activation=torch.tanh)
        self.softmax = nn.Softmax(dim=-1)
        self.decoder_h = nn.Parameter(torch.randn(h_dim), requires_grad=True)

    def forward(self, encoder_h_seq: torch.Tensor, mask: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            encoder_h_seq (:class:`torch.FloatTensor` [batch size, sequence length, dimensions]): Data
                over which to apply the attention mechanism.
            mask (:class:`torch.BoolTensor` [batch size, sequence length]): Mask
                for padded sequences of variable length.

        Returns:
            :class:`tuple` with `output` and `weights`:
            * **output** (:class:`torch.LongTensor` [batch size, output length, dimensions]):
              Tensor containing the attended features.
            * **weights** (:class:`torch.FloatTensor` [batch size, output length, query length]):
              Tensor containing attention weights.
        """
        batch_size, seq_len, h_dim = encoder_h_seq.size()

        encoder_h_seq = self.linear_in(encoder_h_seq.contiguous().view(-1, h_dim))
        encoder_h_seq = encoder_h_seq.view(batch_size, seq_len, h_dim)

        # (batch_size, 1, dimensions) * (batch_size, seq_len, dimensions) -> (batch_size, seq_len)
        attention_scores = torch.bmm(self.decoder_h.expand((batch_size, h_dim)).unsqueeze(1), encoder_h_seq.transpose(1, 2).contiguous())

        # Compute weights across every context sequence
        attention_scores = attention_scores.view(batch_size, -1)
        if mask is not None:
            if mask.dtype is not torch.bool:
                mask = mask.bool()
            attention_scores[~mask] = float("-inf")
        attention_weights = self.softmax(attention_scores)

        # (batch_size, 1, query_len) * (batch_size, query_len, dimensions) -> (batch_size, dimensions)
        output = torch.bmm(attention_weights.unsqueeze(1), encoder_h_seq).squeeze()
        return output, attention_weights

    @staticmethod
    def create_mask(valid_lengths: torch.Tensor, max_len: int = None) -> torch.Tensor:
        if not max_len:
            max_len = valid_lengths.max()
        return torch.arange(max_len, dtype=valid_lengths.dtype, device=valid_lengths.device).expand(len(valid_lengths), max_len) < valid_lengths.unsqueeze(1)


class BertPretrainedClassifier(nn.Module):
    def __init__(self, batch_size: int = 8, dropout: float = 0.1, label_size: int = 2,
                 loss_func: Callable = nn.BCELoss, bert_pretrained_model: str = BERT_PRETRAINED_MODEL,
                 bert_state_dict: str = None, name: str = "OOB", device: torch.device = None):
        super(BertPretrainedClassifier, self).__init__()
        self.name = f"{self.__class__.__name__}-{name}"
        self.batch_size = batch_size
        self.label_size = label_size
        self.dropout = dropout
        self.loss_func = loss_func()
        self.device = device
        self.bert_pretrained_model = bert_pretrained_model
        self.bert_state_dict = bert_state_dict
        self.bert = BertPretrainedClassifier.load_frozen_bert(bert_pretrained_model, bert_state_dict)
        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model,
                                                       do_lower_case=bool(BERT_PRETRAINED_MODEL.endswith("uncased")))
        new_tokens = load_new_tokens()
        self.tokenizer.add_tokens(new_tokens)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.embeddings = self.bert.embeddings.word_embeddings
        self.hidden_size = self.bert.config.hidden_size
        self.pooler = HAN_Attention_Pooler_Layer(self.hidden_size)
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.Softmax(dim=-1)
        self.loss_func_ce = CrossEntropyLoss()

    @staticmethod
    def load_frozen_bert(bert_pretrained_model, bert_state_dict):
        if bert_state_dict:
            print(f"Loading pretrained BERT model from: %s" % bert_state_dict)
            config = BertConfig.from_pretrained(os.path.dirname(bert_state_dict))
            fine_tuned_state_dict = torch.load(bert_state_dict, map_location=torch.device('cpu'))
            bert = BertModel.from_pretrained(bert_pretrained_model, config=config, state_dict=fine_tuned_state_dict)
        else:
            print(f"Loading pretrained BERT model from scratch")
            bert = BertModel.from_pretrained(bert_pretrained_model)
        for p in bert.parameters():
            p.requires_grad = False
        return bert

    def forward(self, input_ids, input_mask, labels, kb_arr):  # kb_arr需要添加到日志解析中去！同时labels是否需要padding以对齐？
        last_hidden_states_seq, _ = self.bert(input_ids, attention_mask=input_mask)
        pooled_seq_vector, attention_weights = self.pooler(last_hidden_states_seq, input_mask)
        kb_emb = self.embeddings(kb_arr)

        u_temp = pooled_seq_vector.unsqueeze(1).expand_as(kb_emb)
        prob_logit = torch.sum(kb_emb*u_temp, dim=2)

        # # Masking pad kb tokens
        # comparison_tensor = torch.ones_like(kb_arr, dtype=torch.int64) * EntityPredictionDataset.PAD_TOKEN_IDX  # Matrix to compare
        # mask = torch.eq(kb_arr, comparison_tensor)  # The mask
        # dummy_scores = torch.ones_like(prob_logit) * -99999.0
        # masked_prob_logit = torch.where(mask, dummy_scores, prob_logit)

        # scores = self.sigmoid(masked_prob_logit)
        scores = self.softmax(prob_logit)
        loss = self.loss_func_ce(prob_logit.view(-1, MAX_KB_ARR_LENGTH), labels.view(-1))
        return loss, prob_logit, attention_weights

    def get_trainable_params(self, recurse: bool = True) -> (List[nn.Parameter], int):
        parameters = list(filter(lambda p: p.requires_grad, self.parameters(recurse)))
        num_trainable_parameters = sum([p.flatten().size(0) for p in parameters])
        return parameters, num_trainable_parameters


class LightningBertPretrainedClassifier(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.bert_classifier = BertPretrainedClassifier(**hparams.bert_params)
        self.softmax = nn.Softmax(dim=-1)

    def parameters(self, recurse: bool = ...):
        return self.bert_classifier.parameters(recurse)

    def configure_optimizers(self):
        parameters_list = self.bert_classifier.get_trainable_params()[0]
        if parameters_list:
            return torch.optim.Adam(parameters_list)
        else:
            return [] # PyTorch Lightning hack for test mode with frozen model

    def forward(self, *args):
        return self.bert_classifier.forward(*args)

    def get_trainable_params(self, recurse: bool = True) -> (List[nn.Parameter], int):
        parameters = list(filter(lambda p: p.requires_grad, self.parameters(recurse)))
        num_trainable_parameters = sum([p.flatten().size(0) for p in parameters])
        return parameters, num_trainable_parameters

    @data_loader
    def train_dataloader(self):
        if not self.training:
            return []
        dataset = EntityPredictionDataset(self.hparams.train_data_path, self.hparams.dev_data_path, self.hparams.test_data_path, self.hparams.dataset, "train",
                                          self.hparams.text_column, self.hparams.label_column,
                                          max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids, kb_arr, turn_ids, sample_ids = batch
        loss, scores, pooler_attention_weights = self.forward(input_ids, input_mask, labels, kb_arr)
        return {"loss": loss, "log": {"batch_num": batch_idx, "train_loss": loss}}

    @data_loader
    def val_dataloader(self):
        if not self.training:
            return []
        dataset = EntityPredictionDataset(self.hparams.train_data_path, self.hparams.dev_data_path, self.hparams.test_data_path, self.hparams.dataset, "dev",
                                          self.hparams.text_column, self.hparams.label_column,
                                          max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def validation_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids, kb_arr, turn_ids, sample_ids = batch
        loss, scores, pooler_attention_weights = self.forward(input_ids, input_mask, labels, kb_arr)
        predictions = torch.argmax(scores, dim=-1)
        correct = predictions.eq(labels.view_as(predictions)).double()
        # batch_size = len(scores)
        # f1, f1_cnt = 0, 0
        # sorted, indices = torch.sort(scores, descending=True)
        # for bt in range(batch_size):
        #     prediction_scores_per_sample, labels_per_sample = scores[bt], labels[bt]
        #     num_gold_ents = torch.sum(labels_per_sample).item()
        #     topk_labels = torch.index_select(labels, 1, indices[bt, :TOP_K])
        #     num_topk_gt_ents = torch.sum(topk_labels[bt]).item()
        #     precision = num_topk_gt_ents / TOP_K
        #     recall = num_topk_gt_ents / num_gold_ents if num_gold_ents != 0 else 0
        #     single_f1 = (2*precision*recall) / float(precision+recall) if float(precision+recall) != 0.0 else 0
        #     f1 += single_f1
        #     f1_cnt += 1
        # F1 = round((f1 / f1_cnt), 5)
        # return {"loss": loss, "log": {"batch_num": batch_idx, "val_loss": loss, "val_accuracy": torch.tensor(F1)}, "progress_bar": {"val_loss": loss, "val_accuracy": torch.tensor(F1)}, "f1": f1, "f1_cnt": f1_cnt}
        return {"loss": loss, "log": {"batch_num": batch_idx, "val_loss": loss, "val_accuracy": correct.mean()}, "progress_bar": {"val_loss": loss, "val_accuracy": correct.mean()}, "correct": correct}

    def validation_end(self, step_outputs):
        total_loss = list()
        # total_f1 = list()
        # total_f1_cnt = list()
        total_correct = list()
        for x in step_outputs:
            total_loss.append(x["loss"])
            # total_f1.append(float(x["f1"]))
            # total_f1_cnt.append(x["f1_cnt"])
            total_correct.append(x["correct"])
        avg_loss = torch.stack(total_loss).mean()
        accuracy = torch.cat(total_correct).mean()
        # f1_sum = torch.sum(torch.tensor(total_f1)).item()
        # f1_cnt_sum = torch.sum(torch.tensor(total_f1_cnt)).item()
        # F1 = round((f1_sum / float(f1_cnt_sum)), 5)
        # print("\nValidation F1 (All Domains): " + str(F1))
        # return {"loss": avg_loss, "progress_bar": {"val_loss": avg_loss, "val_accuracy": torch.tensor(F1)},
        #         "log": {"val_loss": avg_loss, "val_accuracy": torch.tensor(F1)}, "F1": torch.tensor(F1)}
        return {"loss": avg_loss, "progress_bar": {"val_loss": avg_loss, "val_accuracy": accuracy},
                "log": {"val_loss": avg_loss, "val_accuracy": accuracy}}

    @data_loader
    def test_dataloader(self):
        dataset = EntityPredictionDataset(self.hparams.train_data_path, self.hparams.dev_data_path, self.hparams.test_data_path, self.hparams.dataset, "test",
                                          self.hparams.text_column, self.hparams.label_column,
                                          max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def test_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids, kb_arr, turn_ids, sample_ids = batch
        loss, scores, pooler_attention_weights = self.forward(input_ids, input_mask, labels, kb_arr)
        predictions = torch.argmax(scores, dim=-1)
        correct = predictions.eq(labels.view_as(predictions)).double()
        prob_soft = self.softmax(scores)
        step_ret = []
        for bt in range(input_ids.shape[0]):
            sample_id = sample_ids[bt].item()
            turn_id = turn_ids[bt].item()
            prob = prob_soft[bt].tolist()
            ent_label = labels[bt].item()
            step_ret.append({"sample_id": sample_id, "turn_id": turn_id, "prob_soft": prob, "ent_label": ent_label})
        # f1, f1_cnt = 0, 0
        # sorted, indices = torch.sort(scores, descending=True)
        # batch_size = len(scores)
        # for bt in range(batch_size):
        #     prediction_scores_per_sample, labels_per_sample = scores[bt], labels[bt]
        #     num_gold_ents = torch.sum(labels_per_sample).item()
        #     topk_labels = torch.index_select(labels, 1, indices[bt, :TOP_K])
        #     num_topk_gt_ents = torch.sum(topk_labels[bt]).item()
        #     precision = num_topk_gt_ents / TOP_K
        #     recall = num_topk_gt_ents / num_gold_ents if num_gold_ents != 0 else 0
        #     single_f1 = (2*precision*recall) / float(precision+recall) if float(precision+recall) != 0.0 else 0
        #     f1 += single_f1
        #     f1_cnt += 1
        # F1 = round((f1 / f1_cnt), 5)
        # return {"loss": loss, "log": {"batch_num": batch_idx, "test_loss": loss, "test_accuracy": torch.tensor(F1)}, "progress_bar": {"test_loss": loss, "test_accuracy": torch.tensor(F1)}, "f1": f1, "f1_cnt": f1_cnt}
        return {"loss": loss, "log": {"batch_num": batch_idx, "test_loss": loss, "test_accuracy": correct.mean()}, "progress_bar": {"test_loss": loss, "test_accuracy": correct.mean()}, "correct": correct, "step_ret": step_ret}

    def test_end(self, step_outputs):
        total_loss = list()
        # total_f1 = list()
        # total_f1_cnt = list()
        total_correct = list()
        total_ret = list()
        for x in step_outputs:
            total_loss.append(x["loss"])
            # total_f1.append(float(x["f1"]))
            # total_f1_cnt.append(x["f1_cnt"])
            total_correct.append(x["correct"])
            for elm in x["step_ret"]:
                total_ret.append(elm)
        # save_path = "/Users/shiquan/PycharmProjects/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_2.json"
        # save_path = "/home/yimeng/shiquan/deBiasing-Dialogue/Dialogue_Annotator/pipeline/model_outputs/test_outputs_1.json"
        save_path = self.hparams.save_path
        fout = open(save_path, "w")
        total_ret_str = json.dumps(total_ret, indent=4)
        fout.write(total_ret_str)
        fout.close()
        avg_loss = torch.stack(total_loss).mean()
        accuracy = torch.cat(total_correct).mean()
        # f1_sum = torch.sum(torch.tensor(total_f1)).item()
        # f1_cnt_sum = torch.sum(torch.tensor(total_f1_cnt)).item()
        # F1 = round((f1_sum / float(f1_cnt_sum)), 5)
        # print("\nTest F1 (All Domains): " + str(F1))
        # return {"loss": avg_loss,
        #         "progress_bar": {"test_loss": avg_loss, "test_accuracy": torch.tensor(F1)},
        #         "log": {"test_loss": avg_loss, "test_accuracy": torch.tensor(F1)}, "F1": torch.tensor(F1)}
        return {"loss": avg_loss,
                "progress_bar": {"test_loss": avg_loss, "test_accuracy": accuracy},
                "log": {"test_loss": avg_loss, "test_accuracy": accuracy}}


class DebiasedBertPretrainedClassifier(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.classifier_bias = LightningBertPretrainedClassifier.load_from_checkpoint(self.hparams.ckpt_path)
        self.classifier_debias = LightningBertPretrainedClassifier(self.hparams)
        self.froze_classifier_parameters(self.classifier_bias)
        self.loss_func_ce = nn.CrossEntropyLoss()

    def parameters(self, recurse: bool = ...):
        parameter_list_b = [p for p in self.classifier_bias.parameters(recurse)]
        parameter_list_db = [p for p in self.classifier_debias.parameters(recurse)]
        return parameter_list_b + parameter_list_db

    def configure_optimizers(self):
        parameters_list_b = self.classifier_bias.get_trainable_params()[0]
        parameters_list_db = self.classifier_debias.get_trainable_params()[0]
        parameters_list = parameters_list_b + parameters_list_db
        if parameters_list:
            return torch.optim.Adam(parameters_list)
        else:
            return [] # PyTorch Lightning hack for test mode with frozen model

    def froze_classifier_parameters(self, cls):
        for p in cls.parameters():
            p.requires_grad = False
        return

    def forward(self, input_ids, input_mask, labels, kb_arr):
        loss_bias, logits_bias, _ = self.classifier_bias(input_ids, input_mask, labels, kb_arr)
        loss_debias, logits_debias, _ = self.classifier_debias(input_ids, input_mask, labels, kb_arr)
        # logits_final = logits_debias - logits_bias
        logits_final = logits_debias
        loss = self.loss_func_ce(logits_final.view(-1, MAX_KB_ARR_LENGTH), labels.view(-1))
        return loss, logits_final

    @data_loader
    def train_dataloader(self):
        if not self.training:
            return []
        dataset = EntityPredictionDataset(self.hparams.data_path, self.hparams.dataset, "train",
                                          self.hparams.text_column, self.hparams.label_column,
                                          max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def training_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids, kb_arr, turn_ids, sample_ids = batch
        loss, scores = self.forward(input_ids, input_mask, labels, kb_arr)
        return {"loss": loss, "log": {"batch_num": batch_idx, "train_loss": loss}}

    @data_loader
    def val_dataloader(self):
        if not self.training:
            return []
        dataset = EntityPredictionDataset(self.hparams.data_path, self.hparams.dataset, "dev",
                                          self.hparams.text_column, self.hparams.label_column,
                                          max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def validation_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids, kb_arr, turn_ids, sample_ids = batch
        loss, scores = self.forward(input_ids, input_mask, labels, kb_arr)
        predictions = torch.argmax(scores, dim=-1)
        correct = predictions.eq(labels.view_as(predictions)).double()
        return {"loss": loss, "log": {"batch_num": batch_idx, "val_loss": loss, "val_accuracy": correct.mean()}, "progress_bar": {"val_loss": loss, "val_accuracy": correct.mean()}, "correct": correct}

    def validation_end(self, step_outputs):
        total_loss = list()
        total_correct = list()
        for x in step_outputs:
            total_loss.append(x["loss"])
            total_correct.append(x["correct"])
        avg_loss = torch.stack(total_loss).mean()
        accuracy = torch.cat(total_correct).mean()
        return {"loss": avg_loss, "progress_bar": {"val_loss": avg_loss, "val_accuracy": accuracy},
                "log": {"val_loss": avg_loss, "val_accuracy": accuracy}}

    @data_loader
    def test_dataloader(self):
        dataset = EntityPredictionDataset(self.hparams.data_path, self.hparams.dataset, "test",
                                          self.hparams.text_column, self.hparams.label_column,
                                          max_seq_length=self.hparams.max_seq_len)
        dataloader = DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=NUM_CPU)
        return dataloader

    def test_step(self, batch, batch_idx):
        input_ids, input_mask, labels, unique_ids, kb_arr, turn_ids, sample_ids = batch
        loss, scores = self.forward(input_ids, input_mask, labels, kb_arr)
        predictions = torch.argmax(scores, dim=-1)
        correct = predictions.eq(labels.view_as(predictions)).double()
        return {"loss": loss, "log": {"batch_num": batch_idx, "test_loss": loss, "test_accuracy": correct.mean()}, "progress_bar": {"test_loss": loss, "test_accuracy": correct.mean()}, "correct": correct}

    def test_end(self, step_outputs):
        total_loss = list()
        total_correct = list()
        for x in step_outputs:
            total_loss.append(x["loss"])
            total_correct.append(x["correct"])
        avg_loss = torch.stack(total_loss).mean()
        accuracy = torch.cat(total_correct).mean()
        return {"loss": avg_loss,
                "progress_bar": {"test_loss": avg_loss, "test_accuracy": accuracy},
                "log": {"test_loss": avg_loss, "test_accuracy": accuracy}}


class EntityPredictionDataset(Dataset):

    PAD_TOKEN_IDX = 0

    def __init__(self, train_data_path: str, dev_data_path: str, test_data_path: str, dataset: str, subset: str, text_column: str, label_column: str,
                 max_seq_length: int = MAX_SENTIMENT_SEQ_LENGTH, bert_pretrained_model: str = BERT_PRETRAINED_MODEL):
        super(EntityPredictionDataset, self).__init__()
        if subset not in ('train', 'dev', 'test'):
            raise ValueError("subset argument must be {train, dev,test}")
        if dataset == 'multiwoz':
            if subset in ['train']:
                self.dataset_file = f"{train_data_path}"
                # self.dataset_file = f"{data_path}/MultiWOZ_2.2/{subset}/{subset}_utterances_w_kb_w_gold_sm.txt"
            elif subset in ['dev']:
                self.dataset_file = f"{dev_data_path}"
            elif subset in ['test']:
                self.dataset_file = f"{test_data_path}"
                # self.dataset_file = f"{data_path}/MultiWOZ_2.2/{subset}/{subset}_utterances_w_kb_w_gold_sm.txt"
        elif dataset == 'sgd':
            if subset in ['train']:
                self.dataset_file = f"{train_data_path}"
            elif subset in ['dev']:
                self.dataset_file = f"{dev_data_path}"
            elif subset in ['test']:
                self.dataset_file = f"{test_data_path}"
            # self.dataset_file = f"{data_path}/SGD/{subset}/{subset}_utterances_w_kb_w_gold.txt"
        self.subset = subset
        self.text_column = text_column
        self.label_column = label_column
        self.max_seq_len = max_seq_length
        self.tokenizer = BertTokenizer.from_pretrained(bert_pretrained_model,
                                                       do_lower_case=bool(BERT_PRETRAINED_MODEL.endswith("uncased")))
        new_tokens = load_new_tokens()
        print("[ BEFORE ] tokenizer vocab size:", len(self.tokenizer))
        self.tokenizer.add_tokens(new_tokens)
        print("[ AFTER ] tokenizer vocab size:", len(self.tokenizer))

        data = self.read_lang(self.dataset_file)
        features, labels = self.convert_examples_to_features(data)
        self.dataset = self.create_tensor_data(features, labels)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.dataset[item]

    def read_lang(self, path):
        print(("Reading lines from {}".format(path)))
        data, kb_arr = [], []
        dialogue_history = ''
        with open(path) as fin:
            cnt_lin, sample_counter, turn_cnt = 0, 1, 0
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
                        dialogue_history = (dialogue_history + " " + u).strip(" ")

                        tokens = self.tokenizer.tokenize(dialogue_history)
                        tokens = self.trunc_seq(tokens, self.max_seq_len)
                        tokens = tuple([CLS_TOKEN] + tokens + [SEP_TOKEN])

                        # obtain gold entity
                        gold_ent = ast.literal_eval(gold_ent)

                        # obtain gt entity labels
                        # ent_labels = [1 if ent in gold_ent else 0 for ent in kb_arr]  # ent_labels version 1
                        if len(gold_ent) == 0:
                            ent_labels = len(kb_arr)
                        elif len(gold_ent) >= 1:
                            for idx, ent in enumerate(kb_arr):
                                if ent in gold_ent:
                                    ent_labels = idx
                                    break

                        data_detail = {
                            'dialogue_history': tokens,
                            'response': r,
                            'kb_arr': list(kb_arr + [NULL_TOKEN]),
                            'gold_ent': list(gold_ent),
                            'ent_labels': ent_labels,
                            'id': int(turn_cnt),
                            'ID': int(cnt_lin),
                            'domain': task_type
                        }
                        data.append(data_detail)

                        dialogue_history = dialogue_history + " " + r
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
                    kb_arr = []
                    dialogue_history = ''
                    turn_cnt = 0
        return data

    def convert_examples_to_features(self, data):
        features_list = list()
        labels_list = list()
        for i, example in enumerate(data):
            feature = example[self.text_column]
            label = example[self.label_column]  # 需要pad吗？为什么？
            kb_arr = example['kb_arr']
            turn_id = example['id']
            sample_id = example['ID']
            input_ids = self.tokenizer.convert_tokens_to_ids(feature)
            input_mask = [1] * len(input_ids)
            kb_arr_id = self.tokenizer.convert_tokens_to_ids(kb_arr)  # 需要pad吗？为什么？
            while len(input_ids) < self.max_seq_len:
                input_ids.append(self.PAD_TOKEN_IDX)
                input_mask.append(self.PAD_TOKEN_IDX)
            while len(kb_arr_id) < MAX_KB_ARR_LENGTH:
                kb_arr_id.append(self.PAD_TOKEN_IDX)
            # while len(label) < MAX_KB_ARR_LENGTH:
            #     label.append(self.PAD_TOKEN_IDX)
            assert len(input_ids) == self.max_seq_len
            assert len(input_mask) == self.max_seq_len
            assert len(kb_arr_id) == MAX_KB_ARR_LENGTH
            # assert len(label) == MAX_KB_ARR_LENGTH
            features_list.append(InputFeatures(unique_id=example['ID'], tokens=example[self.text_column],
                                               input_ids=input_ids, input_mask=input_mask, kb_arr=kb_arr_id,
                                               turn_id=turn_id, sample_id=sample_id))
            labels_list.append(InputLabel(unique_id=example['ID'], label=label))
        return features_list, labels_list

    def create_tensor_data(self, features, labels):
        input_ids_list = list()
        input_masks_list = list()
        input_unique_id_list = list()
        input_labels_list = list()
        input_kb_arr_list = list()
        input_turn_id_list = list()
        input_sample_id_list = list()
        for f, l in zip(features, labels):
            input_ids_list.append(f.input_ids)
            input_masks_list.append(f.input_mask)
            assert l.unique_id == f.unique_id
            input_unique_id_list.append(f.unique_id)
            input_kb_arr_list.append(f.kb_arr)
            input_labels_list.append(l.label)
            input_turn_id_list.append(f.turn_id)
            input_sample_id_list.append(f.sample_id)
        all_input_ids = torch.tensor(input_ids_list, dtype=torch.long)
        all_input_mask = torch.tensor(input_masks_list, dtype=torch.long)
        all_labels = torch.tensor(input_labels_list, dtype=torch.long)
        all_unique_id = torch.tensor(input_unique_id_list, dtype=torch.long)
        all_kb_arr = torch.tensor(input_kb_arr_list, dtype=torch.long)
        all_turn_id = torch.tensor(input_turn_id_list, dtype=torch.long)
        all_sample_id = torch.tensor(input_sample_id_list, dtype=torch.long)

        return TensorDataset(all_input_ids, all_input_mask, all_labels, all_unique_id, all_kb_arr, all_turn_id, all_sample_id)

    @staticmethod
    def trunc_seq(tokens, max_num_tokens):
        """Truncates a pair of sequences to a maximum sequence length. Lifted from Google's BERT repo."""
        l = 0
        r = len(tokens)
        trunc_tokens = list(tokens)
        while r - l > (max_num_tokens-2):
            # We want to sometimes truncate from the front and sometimes from the
            # back to add more randomness and avoid biases.
            if random() < 0.5:
                l += 1
            else:
                r -= 1
        return trunc_tokens[l:r]

