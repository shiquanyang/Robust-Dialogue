from transformers.modeling_bert import BertLMPredictionHead, BertPreTrainedModel, BertModel
from BERT.lm_finetune.grad_reverse_layer import GradReverseLayerFunction
from BERT.bert_text_dataset import BertTextDataset
from BERT.bert_pos_tagger import BertTokenClassificationDataset
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import torch


class DSTPredictionHead(nn.Module):
    def __init__(self, config, output_size):
        super(DSTPredictionHead, self).__init__()
        self.decoder = nn.Linear(config.hidden_size, output_size)
        self.alpha = 1.

    def forward(self, hidden_state):
        reversed_hidden_state = GradReverseLayerFunction.apply(hidden_state, self.alpha)
        output = self.decoder(reversed_hidden_state)
        return output


class DSTPretrainingHeads(nn.Module):
    def __init__(self, config, lang):
        super(DSTPretrainingHeads, self).__init__()

        self.predictions = BertLMPredictionHead(config)  # import from standard library.

        self.bus_leaveat_predictions = DSTPredictionHead(config, lang.n_state_values['bus-leaveat'])
        self.train_arriveby_predictions = DSTPredictionHead(config, lang.n_state_values['train-arriveby'])
        self.bus_departure_predictions = DSTPredictionHead(config, lang.n_state_values['bus-departure'])
        self.train_departure_predictions = DSTPredictionHead(config, lang.n_state_values['train-departure'])
        self.hotel_internet_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-internet'])
        self.attraction_type_predictions = DSTPredictionHead(config, lang.n_state_values['attraction-type'])
        self.taxi_leaveat_predictions = DSTPredictionHead(config, lang.n_state_values['taxi-leaveat'])
        self.hotel_parking_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-parking'])
        self.train_bookpeople_predictions = DSTPredictionHead(config, lang.n_state_values['train-bookpeople'])
        self.taxi_arriveby_predictions = DSTPredictionHead(config, lang.n_state_values['taxi-arriveby'])
        self.hotel_bookstay_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-bookstay'])
        self.hotel_stars_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-stars'])
        self.hospital_department_predictions = DSTPredictionHead(config, lang.n_state_values['hospital-department'])
        self.hotel_bookday_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-bookday'])
        self.attraction_area_predictions = DSTPredictionHead(config, lang.n_state_values['attraction-area'])
        self.hotel_type_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-type'])
        self.restaurant_area_predictions = DSTPredictionHead(config, lang.n_state_values['restaurant-area'])
        self.restaurant_booktime_predictions = DSTPredictionHead(config, lang.n_state_values['restaurant-booktime'])
        self.hotel_pricerange_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-pricerange'])
        self.restaurant_food_predictions = DSTPredictionHead(config, lang.n_state_values['restaurant-food'])
        self.hotel_area_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-area'])
        self.restaurant_bookday_predictions = DSTPredictionHead(config, lang.n_state_values['restaurant-bookday'])
        self.hotel_bookpeople_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-bookpeople'])
        self.attraction_name_predictions = DSTPredictionHead(config, lang.n_state_values['attraction-name'])
        self.train_destination_predictions = DSTPredictionHead(config, lang.n_state_values['train-destination'])
        self.restaurant_bookpeople_predictions = DSTPredictionHead(config, lang.n_state_values['restaurant-bookpeople'])
        self.bus_destination_predictions = DSTPredictionHead(config, lang.n_state_values['bus-destination'])
        self.restaurant_name_predictions = DSTPredictionHead(config, lang.n_state_values['restaurant-name'])
        self.train_leaveat_predictions = DSTPredictionHead(config, lang.n_state_values['train-leaveat'])
        self.taxi_destination_predictions = DSTPredictionHead(config, lang.n_state_values['taxi-destination'])
        self.hotel_name_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-name'])
        self.restaurant_pricerange_predictions = DSTPredictionHead(config, lang.n_state_values['restaurant-pricerange'])
        self.bus_day_predictions = DSTPredictionHead(config, lang.n_state_values['bus-day'])
        self.taxi_departure_predictions = DSTPredictionHead(config, lang.n_state_values['taxi-departure'])
        self.train_day_predictions = DSTPredictionHead(config, lang.n_state_values['train-day'])

    def forward(self, sequence_output, pooled_output):
        lm_prediction_scores = self.predictions(sequence_output)
        bus_leaveat_prediction_scores = self.bus_leaveat_predictions(pooled_output)
        train_arriveby_prediction_scores = self.train_arriveby_predictions(pooled_output)
        bus_departure_prediction_scores = self.bus_departure_predictions(pooled_output)
        train_departure_prediction_scores = self.train_departure_predictions(pooled_output)
        hotel_internet_prediction_scores = self.hotel_internet_predictions(pooled_output)
        attraction_type_prediction_scores = self.attraction_type_predictions(pooled_output)
        taxi_leaveat_prediction_scores = self.taxi_leaveat_predictions(pooled_output)
        hotel_parking_prediction_scores = self.hotel_parking_predictions(pooled_output)
        train_bookpeople_prediction_scores = self.train_bookpeople_predictions(pooled_output)
        taxi_arriveby_prediction_scores = self.taxi_arriveby_predictions(pooled_output)
        hotel_bookstay_prediction_scores = self.hotel_bookstay_predictions(pooled_output)
        hotel_stars_prediction_scores = self.hotel_stars_predictions(pooled_output)
        hospital_department_prediction_scores = self.hospital_department_predictions(pooled_output)
        hotel_bookday_prediction_scores = self.hotel_bookday_predictions(pooled_output)
        attraction_area_prediction_scores = self.attraction_area_predictions(pooled_output)
        hotel_type_prediction_scores = self.hotel_type_predictions(pooled_output)
        restaurant_area_prediction_scores = self.restaurant_area_predictions(pooled_output)
        restaurant_booktime_prediction_scores = self.restaurant_booktime_predictions(pooled_output)
        hotel_pricerange_prediction_scores = self.hotel_pricerange_predictions(pooled_output)
        restaurant_food_prediction_scores = self.restaurant_food_predictions(pooled_output)
        hotel_area_prediction_scores = self.hotel_area_predictions(pooled_output)
        restaurant_bookday_prediction_scores = self.restaurant_bookday_predictions(pooled_output)
        hotel_bookpeople_prediction_scores = self.hotel_bookpeople_predictions(pooled_output)
        attraction_name_prediction_scores = self.attraction_name_predictions(pooled_output)
        train_destination_prediction_scores = self.train_destination_predictions(pooled_output)
        restaurant_bookpeople_prediction_scores = self.restaurant_bookpeople_predictions(pooled_output)
        bus_destination_prediction_scores = self.bus_destination_predictions(pooled_output)
        restaurant_name_prediction_scores = self.restaurant_name_predictions(pooled_output)
        train_leaveat_prediction_scores = self.train_leaveat_predictions(pooled_output)
        taxi_destination_prediction_scores = self.taxi_destination_predictions(pooled_output)
        hotel_name_prediction_scores = self.hotel_name_predictions(pooled_output)
        restaurant_pricerange_prediction_scores = self.restaurant_pricerange_predictions(pooled_output)
        bus_day_prediction_scores = self.bus_day_predictions(pooled_output)
        taxi_departure_prediction_scores = self.taxi_departure_predictions(pooled_output)
        train_day_prediction_scores = self.train_day_predictions(pooled_output)
        return (lm_prediction_scores,
                bus_leaveat_prediction_scores,
                train_arriveby_prediction_scores,
                bus_departure_prediction_scores,
                train_departure_prediction_scores,
                hotel_internet_prediction_scores,
                attraction_type_prediction_scores,
                taxi_leaveat_prediction_scores,
                hotel_parking_prediction_scores,
                train_bookpeople_prediction_scores,
                taxi_arriveby_prediction_scores,
                hotel_bookstay_prediction_scores,
                hotel_stars_prediction_scores,
                hospital_department_prediction_scores,
                hotel_bookday_prediction_scores,
                attraction_area_prediction_scores,
                hotel_type_prediction_scores,
                restaurant_area_prediction_scores,
                restaurant_booktime_prediction_scores,
                hotel_pricerange_prediction_scores,
                restaurant_food_prediction_scores,
                hotel_area_prediction_scores,
                restaurant_bookday_prediction_scores,
                hotel_bookpeople_prediction_scores,
                attraction_name_prediction_scores,
                train_destination_prediction_scores,
                restaurant_bookpeople_prediction_scores,
                bus_destination_prediction_scores,
                restaurant_name_prediction_scores,
                train_leaveat_prediction_scores,
                taxi_destination_prediction_scores,
                hotel_name_prediction_scores,
                restaurant_pricerange_prediction_scores,
                bus_day_prediction_scores,
                taxi_departure_prediction_scores,
                train_day_prediction_scores)


class BertForDSTPretraining(BertPreTrainedModel):
    def __init__(self, config, kw=None):
        super(BertForDSTPretraining, self).__init__(config)
        self.lang = kw

        self.bert = BertModel(config)
        self.cls = DSTPretrainingHeads(config, self.lang)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.cls(sequence_output, pooled_output)
        outputs = prediction_scores + outputs[2:]
        return outputs


class UserIntentPredictionHead(nn.Module):
    def __init__(self, config):
        super(UserIntentPredictionHead, self).__init__()
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, hidden_state):
        output = self.classifier(hidden_state)
        # output = self.softmax(output)
        return output


class PretrainingHeadswControl(nn.Module):
    def __init__(self, config, lang):
        super(PretrainingHeadswControl, self).__init__()
        self.lang = lang

        self.predictions = BertLMPredictionHead(config)  # import from standard library.

        self.bus_leaveat_predictions = DSTPredictionHead(config, lang.n_state_values['bus-leaveat'])
        self.train_arriveby_predictions = DSTPredictionHead(config, lang.n_state_values['train-arriveby'])
        self.bus_departure_predictions = DSTPredictionHead(config, lang.n_state_values['bus-departure'])
        self.train_departure_predictions = DSTPredictionHead(config, lang.n_state_values['train-departure'])
        self.hotel_internet_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-internet'])
        self.attraction_type_predictions = DSTPredictionHead(config, lang.n_state_values['attraction-type'])
        self.taxi_leaveat_predictions = DSTPredictionHead(config, lang.n_state_values['taxi-leaveat'])
        self.hotel_parking_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-parking'])
        self.train_bookpeople_predictions = DSTPredictionHead(config, lang.n_state_values['train-bookpeople'])
        self.taxi_arriveby_predictions = DSTPredictionHead(config, lang.n_state_values['taxi-arriveby'])
        self.hotel_bookstay_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-bookstay'])
        self.hotel_stars_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-stars'])
        self.hospital_department_predictions = DSTPredictionHead(config, lang.n_state_values['hospital-department'])
        self.hotel_bookday_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-bookday'])
        self.attraction_area_predictions = DSTPredictionHead(config, lang.n_state_values['attraction-area'])
        self.hotel_type_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-type'])
        self.restaurant_area_predictions = DSTPredictionHead(config, lang.n_state_values['restaurant-area'])
        self.restaurant_booktime_predictions = DSTPredictionHead(config, lang.n_state_values['restaurant-booktime'])
        self.hotel_pricerange_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-pricerange'])
        self.restaurant_food_predictions = DSTPredictionHead(config, lang.n_state_values['restaurant-food'])
        self.hotel_area_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-area'])
        self.restaurant_bookday_predictions = DSTPredictionHead(config, lang.n_state_values['restaurant-bookday'])
        self.hotel_bookpeople_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-bookpeople'])
        self.attraction_name_predictions = DSTPredictionHead(config, lang.n_state_values['attraction-name'])
        self.train_destination_predictions = DSTPredictionHead(config, lang.n_state_values['train-destination'])
        self.restaurant_bookpeople_predictions = DSTPredictionHead(config, lang.n_state_values['restaurant-bookpeople'])
        self.bus_destination_predictions = DSTPredictionHead(config, lang.n_state_values['bus-destination'])
        self.restaurant_name_predictions = DSTPredictionHead(config, lang.n_state_values['restaurant-name'])
        self.train_leaveat_predictions = DSTPredictionHead(config, lang.n_state_values['train-leaveat'])
        self.taxi_destination_predictions = DSTPredictionHead(config, lang.n_state_values['taxi-destination'])
        self.hotel_name_predictions = DSTPredictionHead(config, lang.n_state_values['hotel-name'])
        self.restaurant_pricerange_predictions = DSTPredictionHead(config, lang.n_state_values['restaurant-pricerange'])
        self.bus_day_predictions = DSTPredictionHead(config, lang.n_state_values['bus-day'])
        self.taxi_departure_predictions = DSTPredictionHead(config, lang.n_state_values['taxi-departure'])
        self.train_day_predictions = DSTPredictionHead(config, lang.n_state_values['train-day'])

        self.user_intent_predictions = UserIntentPredictionHead(config)

    def forward(self, sequence_output, pooled_output):
        lm_prediction_scores = self.predictions(sequence_output)
        bus_leaveat_prediction_scores = self.bus_leaveat_predictions(pooled_output)
        train_arriveby_prediction_scores = self.train_arriveby_predictions(pooled_output)
        bus_departure_prediction_scores = self.bus_departure_predictions(pooled_output)
        train_departure_prediction_scores = self.train_departure_predictions(pooled_output)
        hotel_internet_prediction_scores = self.hotel_internet_predictions(pooled_output)
        attraction_type_prediction_scores = self.attraction_type_predictions(pooled_output)
        taxi_leaveat_prediction_scores = self.taxi_leaveat_predictions(pooled_output)
        hotel_parking_prediction_scores = self.hotel_parking_predictions(pooled_output)
        train_bookpeople_prediction_scores = self.train_bookpeople_predictions(pooled_output)
        taxi_arriveby_prediction_scores = self.taxi_arriveby_predictions(pooled_output)
        hotel_bookstay_prediction_scores = self.hotel_bookstay_predictions(pooled_output)
        hotel_stars_prediction_scores = self.hotel_stars_predictions(pooled_output)
        hospital_department_prediction_scores = self.hospital_department_predictions(pooled_output)
        hotel_bookday_prediction_scores = self.hotel_bookday_predictions(pooled_output)
        attraction_area_prediction_scores = self.attraction_area_predictions(pooled_output)
        hotel_type_prediction_scores = self.hotel_type_predictions(pooled_output)
        restaurant_area_prediction_scores = self.restaurant_area_predictions(pooled_output)
        restaurant_booktime_prediction_scores = self.restaurant_booktime_predictions(pooled_output)
        hotel_pricerange_prediction_scores = self.hotel_pricerange_predictions(pooled_output)
        restaurant_food_prediction_scores = self.restaurant_food_predictions(pooled_output)
        hotel_area_prediction_scores = self.hotel_area_predictions(pooled_output)
        restaurant_bookday_prediction_scores = self.restaurant_bookday_predictions(pooled_output)
        hotel_bookpeople_prediction_scores = self.hotel_bookpeople_predictions(pooled_output)
        attraction_name_prediction_scores = self.attraction_name_predictions(pooled_output)
        train_destination_prediction_scores = self.train_destination_predictions(pooled_output)
        restaurant_bookpeople_prediction_scores = self.restaurant_bookpeople_predictions(pooled_output)
        bus_destination_prediction_scores = self.bus_destination_predictions(pooled_output)
        restaurant_name_prediction_scores = self.restaurant_name_predictions(pooled_output)
        train_leaveat_prediction_scores = self.train_leaveat_predictions(pooled_output)
        taxi_destination_prediction_scores = self.taxi_destination_predictions(pooled_output)
        hotel_name_prediction_scores = self.hotel_name_predictions(pooled_output)
        restaurant_pricerange_prediction_scores = self.restaurant_pricerange_predictions(pooled_output)
        bus_day_prediction_scores = self.bus_day_predictions(pooled_output)
        taxi_departure_prediction_scores = self.taxi_departure_predictions(pooled_output)
        train_day_prediction_scores = self.train_day_predictions(pooled_output)
        user_intent_prediction_scores = self.user_intent_predictions(pooled_output)

        return (lm_prediction_scores,
                bus_leaveat_prediction_scores,
                train_arriveby_prediction_scores,
                bus_departure_prediction_scores,
                train_departure_prediction_scores,
                hotel_internet_prediction_scores,
                attraction_type_prediction_scores,
                taxi_leaveat_prediction_scores,
                hotel_parking_prediction_scores,
                train_bookpeople_prediction_scores,
                taxi_arriveby_prediction_scores,
                hotel_bookstay_prediction_scores,
                hotel_stars_prediction_scores,
                hospital_department_prediction_scores,
                hotel_bookday_prediction_scores,
                attraction_area_prediction_scores,
                hotel_type_prediction_scores,
                restaurant_area_prediction_scores,
                restaurant_booktime_prediction_scores,
                hotel_pricerange_prediction_scores,
                restaurant_food_prediction_scores,
                hotel_area_prediction_scores,
                restaurant_bookday_prediction_scores,
                hotel_bookpeople_prediction_scores,
                attraction_name_prediction_scores,
                train_destination_prediction_scores,
                restaurant_bookpeople_prediction_scores,
                bus_destination_prediction_scores,
                restaurant_name_prediction_scores,
                train_leaveat_prediction_scores,
                taxi_destination_prediction_scores,
                hotel_name_prediction_scores,
                restaurant_pricerange_prediction_scores,
                bus_day_prediction_scores,
                taxi_departure_prediction_scores,
                train_day_prediction_scores,
                user_intent_prediction_scores)


class BertForDSTwControlPreTraining(BertPreTrainedModel):
    def __init__(self, config, kw=None):
        super(BertForDSTwControlPreTraining, self).__init__(config)
        self.lang = kw

        self.bert = BertModel(config)
        self.cls = PretrainingHeadswControl(config, self.lang)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(self.cls.predictions.decoder,
                                   self.bert.embeddings.word_embeddings)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None):
        outputs = self.bert(input_ids,
                           attention_mask=attention_mask,
                           token_type_ids=token_type_ids,
                           position_ids=position_ids,
                           head_mask=head_mask)
        sequence_output, pooled_output = outputs[:2]
        prediction_scores = self.cls(sequence_output, pooled_output)
        outputs = prediction_scores + outputs[2:]
        return outputs