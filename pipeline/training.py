import torch
import json
from pathlib import Path
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
from constants import DIALOGUE_ANNOTATOR_DATASETS_DIR, DIALOGUE_ANNOTATOR_PRETRAIN_DATA_DIR, BERT_PRETRAINED_MODEL, \
    DIALOGUE_EXPERIMENTS_DIR, MAX_SENTIMENT_SEQ_LENGTH, DIALOGUE_ANNOTATOR_PRETRAIN_DST_DIR
from Dialogue_Annotator.pipeline.bert_classifier import LightningBertPretrainedClassifier, LightningHyperparameters, BertPretrainedClassifier
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from argparse import ArgumentParser
from utils import init_logger
from Sentiment_Adjectives.pipeline.predict import predict_models, print_final_metrics

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

### Constants
BATCH_SIZE = 16
# BATCH_SIZE = 8
ACCUMULATE = 1
DROPOUT = 0.1
EPOCHS = 50
# EPOCHS = 1
FP16 = False


def downstream_task_finetuning(args, bert_state_dict):
    data_file = args.pregenerated_data / f"{BERT_PRETRAINED_MODEL}_epoch_0.json"
    metrics_file = args.pregenerated_data / f"{BERT_PRETRAINED_MODEL}_epoch_0_metrics.json"
    assert data_file.is_file() and metrics_file.is_file()
    metrics = json.loads(metrics_file.read_text())
    num_samples = metrics['num_training_examples']

    hparams = {
        "train_data_path": args.train_data_path,
        "dev_data_path": args.dev_data_path,
        "test_data_path": args.test_data_path,
        "save_path": args.save_path,
        # "data_path": DIALOGUE_ANNOTATOR_DATASETS_DIR,
        "dataset": args.dataset,
        "treatment": "DST",
        "masking_method": args.masking_method,
        "pretrain_control": args.pretrained_control,
        "text_column": "dialogue_history",
        "label_column": "ent_labels",
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "accumulate": ACCUMULATE,
        "num_training_examples": num_samples,
        "max_seq_len": MAX_SENTIMENT_SEQ_LENGTH,
        "name": f"Entity_Prediction",
        "bert_params": {
            "dropout": DROPOUT,
            "bert_state_dict": bert_state_dict,
            "name": f"Entity_Prediction"
        }
    }

    print(f"Training {hparams['name']} models")

    if hparams["bert_params"]["bert_state_dict"]:
        if args.pretrained_control:
            hparams["bert_params"]["name"] = f"Entity_Prediction_From_DST_UI_Treated"
            model_save_path = f"entity-prediction-models-from-DST-UI-Treated"
        else:
            hparams["bert_params"]["name"] = f"Entity_Prediction_From_DST_Treated"
            model_save_path = f"entity-prediction-models-from-DST-Treated"
    else:
        hparams["bert_params"]["name"] = f"Entity_Prediction_From_Scratch"
        model_save_path = f"entity-prediction-models-from-scratch"

    OUTPUT_DIR = f"{DIALOGUE_EXPERIMENTS_DIR}/{hparams['treatment']}/{hparams['bert_params']['name']}"
    checkpoint_callback = ModelCheckpoint(
        monitor='val_accuracy',
        filepath=model_save_path,
        prefix='entity-prediction',
        mode='max',
        save_best_only=False
    )
    early_stopping_callback = EarlyStopping('val_accuracy', patience=2, mode='max')
    trainer = Trainer(gpus=1 if DEVICE.type == "cuda" else 0,
                      default_save_path=OUTPUT_DIR,
                      show_progress_bar=True,
                      accumulate_grad_batches=hparams["accumulate"],
                      max_nb_epochs=hparams["epochs"],
                      early_stop_callback=early_stopping_callback,
                      checkpoint_callback=checkpoint_callback)
    hparams['output_path'] = trainer.logger.experiment.log_dir.rstrip('tf')
    logger = init_logger("training", hparams['output_path'])
    logger.info(f"Training Entity_Prediction for {hparams['epochs']} epochs")
    hparams["bert_params"]["batch_size"] = hparams["batch_size"]
    model = LightningBertPretrainedClassifier(LightningHyperparameters(hparams))
    trainer.fit(model)
    trainer.test()
    print_final_metrics(hparams["bert_params"]["name"], trainer.tqdm_metrics, logger)


def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=False)
    parser.add_argument("--dataset", type=str, default="multiwoz", choices=("multiwoz","sgd"),
                        help="Specify dataset for experiments")
    parser.add_argument("--group", type=str, default="F", choices=("F", "CF"),
                        help="Specify data group for experiments: F (factual) or CF (counterfactual)")
    parser.add_argument("--masking_method", type=str, default="mlm_prob", choices=("double_num_adj", "mlm_prob"),
                        help="Method of determining num masked tokens in sentence")
    parser.add_argument("--pretrained_epoch", type=int, default=0,
                        help="Specify epoch for pretrained models: 0-4")
    parser.add_argument("--pretrained_control", action="store_true",
                        help="Use pretraining model with control task")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of epochs to train for")
    parser.add_argument("--init_using_pretrained_bert", action="store_true",
                        help="Whether use task-specific pretrained bert initialization or train from scratch")
    parser.add_argument("--train_data_path", type=str,
                        help="Specify train dataset path for experiments")
    parser.add_argument("--test_data_path", type=str,
                        help="Specify test dataset path for experiments")
    parser.add_argument("--dev_data_path", type=str,
                        help="Specify development dataset path for experiments")
    parser.add_argument("--save_path", type=str,
                        help="Specify testing results saving path for experiments")
    args = parser.parse_args()

    args.pregenerated_data = Path(DIALOGUE_ANNOTATOR_PRETRAIN_DATA_DIR) / args.masking_method

    if args.pretrained_control:
        pretrained_model_dir = f"{DIALOGUE_ANNOTATOR_PRETRAIN_DST_DIR}/{args.masking_method}/model_control"
    else:
        pretrained_model_dir = f"{DIALOGUE_ANNOTATOR_PRETRAIN_DST_DIR}/{args.masking_method}/model"

    if args.pretrained_epoch is not None:
        pretrained_model_dir = f"{pretrained_model_dir}/epoch_{args.pretrained_epoch}"

    if args.init_using_pretrained_bert:
        bert_state_dict = f"{pretrained_model_dir}/pytorch_model.bin"
    else:
        bert_state_dict = None

    downstream_task_finetuning(args, bert_state_dict)


if __name__ == '__main__':
    main()