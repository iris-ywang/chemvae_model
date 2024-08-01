import numpy as np
from itertools import chain

from .pairwise_data import PairwiseDataInfo, PairwiseValues
from .pa_basics.all_pairs import pair_by_pair_id_per_feature
# from .pa_basics.rating import rating_elo

class PairwiseRegModel():

    def __init__(self,
                 pairwise_data_info: PairwiseDataInfo,
                 ML_reg=None,
                 pairing_method=pair_by_pair_id_per_feature,
                 search_model=None,
                 batch_size=1000000,
                 pairing_params=None):
        self.pairwise_data_info = pairwise_data_info
        self.ML_reg = ML_reg
        self.batch_size = batch_size
        self.pairing_params = pairing_params
        self.pairing_method = pairing_method
        self.search_model = search_model
        self.Y_values = PairwiseValues()

        self.trained_reg_model = None

    def fit(self):
        train_pairs = self.pairing_method(
            data=self.pairwise_data_info.train_test,
            pair_ids=self.pairwise_data_info.c1_test_pair_ids
        )
        self.Y_values.Y_pa_c1_true = list(train_pairs[:, 0])


        trained_reg_model, _ = build_ml_model(
            model=self.ML_reg,
            train_data=train_pairs,
            search_model=self.search_model,
            test_data=None
        )
        self.trained_reg_model = trained_reg_model
        self.Y_values.Y_pa_c1_nume = list(train_pairs[:, 0])

        return self

    def predict(self):

        if self.Y_values.Y_pa_c2_nume is None:
            self.Y_values.Y_pa_c2_nume_true, self.Y_values.Y_pa_c2_nume = \
                self._fit_dist(self.pairwise_data_info.c2_test_pair_ids)

        if self.Y_values.Y_pa_c3_nume is None:
            self.Y_values.Y_pa_c3_nume_true, self.Y_values.Y_pa_c3_nume = \
                self._fit_dist(self.pairwise_data_info.c3_test_pair_ids)

        return self.Y_values

        # y_ranking_score_test = self.rank(
        #     ranking_method=ranking_method,
        #     ranking_input_type=ranking_input_type,
        #     if_sbbr_dist=if_sbbr_dist
        # )

        # return y_ranking_score_test

    # def rank(self, ranking_method, ranking_input_type, if_sbbr_dist=False):
    #     """ranking_inputs: sub-list of ['c2', 'c3', 'c2_c3', 'c1_c2_c3']"""
    #     combi_types = ranking_input_type.split("_")
    #     Y, test_pair_ids = [], []
    #     for pair_type in combi_types:
    #
    #         if not if_sbbr_dist:
    #             Y += list(getattr(self.Y_values, f"Y_pa_{pair_type}_sign"))
    #         else:
    #             assert self.trained_reg_model is not None
    #             Y += list(
    #                 getattr(self.Y_values, f"Y_pa_{pair_type}_nume")
    #             )
    #         test_pair_ids += getattr(self.pairwise_data_info, f"{pair_type}_test_pair_ids")
    #
    #     y_ranking_score_all = ranking_method(
    #         Y=Y,
    #         test_pair_ids=test_pair_ids,
    #         y_true=self.pairwise_data_info.y_true_all)
    #     y_ranking_score_test = y_ranking_score_all[self.pairwise_data_info.test_ids]
    #
    #     setattr(self, f"y_rank_via_{ranking_input_type}", y_ranking_score_test)
    #     return y_ranking_score_all


    def _fit_dist(self, test_pair_ids):
        number_test_batches = len(test_pair_ids) // self.batch_size
        if number_test_batches < 1: number_test_batches = 0
        Y_pa_dist = []
        Y_pa_true = []
        for test_batch in range(number_test_batches + 1):
            if test_batch != number_test_batches + 1:
                test_pair_id_batch = test_pair_ids[
                                     test_batch * self.batch_size: (test_batch + 1) * self.batch_size]
            else:
                test_pair_id_batch = test_pair_ids[test_batch * self.batch_size:]
            test_pairs_batch = pair_by_pair_id_per_feature(data=self.pairwise_data_info.train_test,
                                                           pair_ids=test_pair_id_batch)
            Y_pa_true += list(test_pairs_batch[:, 0])
            Y_pa_dist += list(self.trained_reg_model.predict(test_pairs_batch[:, 1:]))
            if (test_batch + 1) * self.batch_size >= len(test_pair_ids): break
        return Y_pa_true, Y_pa_dist


def build_ml_model(model, train_data, search_model=None, test_data=None):
    x_train = train_data[:, 1:]
    y_train = train_data[:, 0]

    if search_model is not None:
        search_model.predict(x_train, y_train)
        model = search_model.best_estimator_

    fitted_model = model.fit(x_train, y_train)

    if type(test_data) == np.ndarray:
        x_test = test_data[:, 1:]
        y_test_pred = fitted_model.predict(x_test)
        return fitted_model, y_test_pred
    else:
        return fitted_model, None