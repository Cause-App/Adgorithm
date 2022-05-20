from dataclasses import dataclass
import multiprocessing
import pandas as pd
import numpy as np
import threading
import warnings
import models
import bisect
import tools
import queue
import time

use_multiprocessing = True

Lock = multiprocessing.Lock if use_multiprocessing else threading.Lock
Semaphore = multiprocessing.Semaphore if use_multiprocessing else threading.Semaphore
Queue = multiprocessing.Queue if use_multiprocessing else queue.Queue

rating_to_exp_ctr = {
    1: 0.014925,
    2: 0.024786,
    3: 0.031071,
    4: 0.040562,
    5: 0.068341
}

data = {}


@dataclass
class RankedAd:
    ad_id: int
    value: float

    def __lt__(self, other):
        return self.value < other.value

    def __repr__(self):
        return f"{self.ad_id}{{{self.value}}}"


class Memory:
    def __init__(self, size=100):
        self.size = size
        self.ad_ftrs = (
            np.zeros((self.size, data["ad_ftrs"][0].shape[-1]+1)),
            np.zeros((self.size, *data["ad_ftrs"][1].shape[1:]))
        )
        self.rewards = np.zeros((self.size))
        self.ptr = 0
        self.full = False

    def store(self, ad_ftrs, num_clicks, reward):
        self.ad_ftrs[0][self.ptr][:-1] = ad_ftrs[0]
        self.ad_ftrs[0][self.ptr][-1] = num_clicks
        self.ad_ftrs[1][self.ptr] = ad_ftrs[1]
        self.rewards[self.ptr] = reward

        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True
            self.ptr = 0

    def get_training_data(self):
        if self.full:
            return (self.ad_ftrs, self.rewards)
        else:
            return ((self.ad_ftrs[0][:self.ptr], self.ad_ftrs[1][:self.ptr]), self.rewards[:self.ptr])


class User:
    # Change this number based on how much memory you have available
    all_exp_ctr_sem = Semaphore(10)
    # Change this number based on how much memory you have available
    fit_sem = Semaphore(10)

    def __init__(self, ftrs):
        self.ftrs = ftrs
        self.rl_model = models.create_rl_model(
            data["ad_ftrs"], data["rl_best_hps"])
        self.sorted_ads = []
        self.memory = Memory()
        self.clicks = {}
        self.interaction_history = []
        self.diminishing_returns_coeff = 0.9
        self.ad_memory_size = 100
        self.random_mean_ctrs = []
        self.cheater_ctrs = []
        self.sum_selected_ctrs = 0
        self.interaction_history_max_size = 100
        self.user_model = models.create_uai_model(
            data["ad_ftrs"], data["user_ftrs"], data["ratings"], data["best_hps"])
        self.user_model.load_weights(
            "user_ad_interaction_model_logs/checkpoints/model-000020-1.202267.hdf5")

    def get_all_true_exp_ctr(self, ad_ids, ad_ftrs):
        inputs = (
            np.concatenate([np.broadcast_to(self.ftrs[np.newaxis], (len(
                ad_ftrs[0]), *self.ftrs.shape)), ad_ftrs[0]], axis=-1),
            ad_ftrs[1]
        )
        with self.__class__.all_exp_ctr_sem:
            outputs = self.user_model(inputs, training=False)
        ratings = np.argmax(outputs, axis=-1) + 1
        exp_ctrs = np.zeros_like(ratings, dtype="float32")
        for i, rating in enumerate(ratings):
            exp_ctrs[i] = rating_to_exp_ctr[rating]
            if (ad_id := ad_ids[i]) in self.clicks:
                exp_ctrs[i] *= self.diminishing_returns_coeff ** self.clicks[ad_id]

        return exp_ctrs

    def get_true_exp_ctr(self, ad_id, ad_ftrs):
        return self.get_all_true_exp_ctr([ad_id], (ad_ftrs[0][np.newaxis], ad_ftrs[1][np.newaxis]))[0]

    def predict_reward(self, ad_id, ad_ftrs):
        c = self.clicks[ad_id] if ad_id in self.clicks else 0
        # No semaphore is required here because the inputs are small
        exp_reward = self.rl_model((np.concatenate((ad_ftrs[0], [c]))[
                                   np.newaxis], ad_ftrs[1][np.newaxis]))[0, 0]
        ra = RankedAd(ad_id, exp_reward)
        for i, existing_ad in enumerate(self.sorted_ads):
            if existing_ad.ad_id == ad_id:
                del self.sorted_ads[i]
                break
        bisect.insort(self.sorted_ads, ra)

    def interact_with(self, ad_id, ad_ftrs):
        exp_ctr = self.get_true_exp_ctr(ad_id, ad_ftrs)
        self.sum_selected_ctrs += exp_ctr
        did_click = np.random.uniform() <= exp_ctr
        self.interaction_history.append(did_click)
        if len(self.interaction_history) > self.interaction_history_max_size:
            self.interaction_history = self.interaction_history[1:]
        if did_click:
            if ad_id not in self.clicks:
                self.clicks[ad_id] = 1
            else:
                self.clicks[ad_id] += 1
            AdDatabase.ad_was_clicked(ad_id)
        return did_click

    def select_important_ad(self, p_random=0.1):
        if np.random.uniform() <= p_random or len(self.sorted_ads) == 0:
            return AdDatabase.get_random_ad()
        ad_id = self.sorted_ads[np.random.randint(
            0, len(self.sorted_ads))].ad_id
        return AdDatabase.get_ad(ad_id)

    def select_ad(self, p_random=0.1):
        if np.random.uniform() <= p_random or len(self.sorted_ads) == 0:
            return AdDatabase.get_random_ad()

        return AdDatabase.get_ad(self.sorted_ads[-1].ad_id)

    def remember(self, ad_id, ad_ftrs, reward):
        self.memory.store(
            ad_ftrs, self.clicks[ad_id] if ad_id in self.clicks else 0, reward)
        all_ctrs = self.get_all_true_exp_ctr(
            *AdDatabase.get_all_available_ads())
        self.random_mean_ctrs.append(np.mean(all_ctrs))
        if len(self.random_mean_ctrs) > self.interaction_history_max_size:
            self.random_mean_ctrs = self.random_mean_ctrs[1:]
        self.cheater_ctrs.append(np.max(all_ctrs))
        if len(self.cheater_ctrs) > self.interaction_history_max_size:
            self.cheater_ctrs = self.cheater_ctrs[1:]

    def learn(self, rl_epochs=10, n_ads_to_fetch=10, p_random=0.1):
        if p_random >= 1:
            return
        with self.__class__.fit_sem:
            self.rl_model.fit(
                *self.memory.get_training_data(),
                verbose=0,
                epochs=rl_epochs,
                batch_size=32
            )

        ads_resampled = []
        if len(self.sorted_ads) > 0:
            self.predict_reward(
                *AdDatabase.get_ad(best_ad_id := self.sorted_ads[-1].ad_id))
            ads_resampled.append(best_ad_id)
        while len(ads_resampled) < n_ads_to_fetch:
            ad = self.select_important_ad(p_random=p_random)
            if (ad_id := ad[0]) in ads_resampled:
                continue
            self.predict_reward(*ad)
            ads_resampled.append(ad_id)
        self.sorted_ads = self.sorted_ads[-self.ad_memory_size:]

    def evaluate_prediction_accuracy_for_ads(self, ad_ids, ad_ftrs):
        ground_truth = self.get_all_true_exp_ctr(ad_ids, ad_ftrs)
        num_times_clicked = np.zeros((len(ad_ids), 1))
        for i, ad_id in enumerate(ad_ids):
            num_times_clicked[i] = self.clicks[ad_id] if ad_id in self.clicks else 0
        pred = self.rl_model((
            np.concatenate([ad_ftrs[0], num_times_clicked], axis=-1),
            ad_ftrs[1]
        ), training=False)[:, 0].numpy()
        err = pred - ground_truth
        mse = np.mean(err*err)
        return mse

    def evaluate_prediction_accuracy_for_available_ads(self):
        # Quick and dirty way to detect whether user has p_random=1 or is Cheater
        if len(self.sorted_ads) == 0:
            return np.nan
        return self.evaluate_prediction_accuracy_for_ads(*AdDatabase.get_all_available_ads())

    def evaluate_prediction_accuracy_for_ads_in_memory(self):
        ad_ids = list(x.ad_id for x in self.sorted_ads)
        ad_ftrs = list(AdDatabase.get_ad(
            ad_id, random_if_unavailable=False) for ad_id in ad_ids)
        ad_ftrs = list(x for x in ad_ftrs if x[0] is not None)
        if len(ad_ftrs) == 0:
            return np.nan
        ad_ids, ad_ftrs = list(zip(*ad_ftrs))

        ad_ftrs_0 = list(x[0] for x in ad_ftrs)
        ad_ftrs_1 = list(x[1] for x in ad_ftrs)
        ad_ftrs = (
            np.stack(ad_ftrs_0),
            np.stack(ad_ftrs_1)
        )

        return self.evaluate_prediction_accuracy_for_ads(ad_ids, ad_ftrs)


class Cheater(User):
    def select_ad(self, *args, **kwargs):
        ad_ids, ad_ftrs = AdDatabase.get_all_available_ads()
        winner = np.argmax(self.get_all_true_exp_ctr(ad_ids, ad_ftrs))
        return ad_ids[winner], (ad_ftrs[0][winner], ad_ftrs[1][winner])

    def learn(self, *args, **kwargs):
        return


available_ads = {}
ad_clicks = {}
click_thresh = 100


def set_data(new_data):
    for k, v in new_data.items():
        data[k] = v
    for i, af0 in enumerate(data["ad_ftrs"][0]):
        af1 = data["ad_ftrs"][1][i]
        available_ads[i] = (af0, af1)
        ad_clicks[i] = 0
        data["ad_id_ctr"] = len(data["ad_ftrs"][0])


class NullLock:
    def __enter__(self, *args, **kwargs):
        pass

    def __exit__(self, *args, **kwargs):
        pass


class BypassableLock:
    def __init__(self):
        self.lock = Lock()

    def __call__(self, use_lock=True):
        return self.lock if use_lock else NullLock()

    def __enter__(self, *args, **kwargs):
        return self.lock.__enter__(*args, **kwargs)

    def __exit__(self, *args, **kwargs):
        return self.lock.__exit__(*args, **kwargs)


ad_db_lock = BypassableLock()


class AdDatabase:
    @staticmethod
    def get_ad(ad_id, use_lock=True, random_if_unavailable=True):
        with ad_db_lock(use_lock):
            if ad_id in available_ads:
                return ad_id, available_ads[ad_id]
            return AdDatabase.get_random_ad(use_lock=False) if random_if_unavailable else (None, None)

    @staticmethod
    def get_random_ad(use_lock=True):
        with ad_db_lock(use_lock):
            ad_id = np.random.choice(list(available_ads.keys()))
            return AdDatabase.get_ad(ad_id, use_lock=False)

    @staticmethod
    def ad_was_clicked(ad_id, use_lock=True):
        with ad_db_lock(use_lock):
            ad_clicks[ad_id] += 1
            if ad_clicks[ad_id] >= click_thresh:
                available_ads[data["ad_id_ctr"]] = available_ads[ad_id]
                ad_clicks[data["ad_id_ctr"]] = 0
                data["ad_id_ctr"] += 1
                del ad_clicks[ad_id]
                del available_ads[ad_id]

    @staticmethod
    def get_all_available_ads():
        ad_ids = list(available_ads.keys())
        ad_ftrs = list(available_ads.values())
        ad_ftrs_0 = list(x[0] for x in ad_ftrs)
        ad_ftrs_1 = list(x[1] for x in ad_ftrs)
        ad_ftrs = (
            np.stack(ad_ftrs_0),
            np.stack(ad_ftrs_1)
        )
        return ad_ids, ad_ftrs


class Simulation:
    EVENT_INTERACTION = 0
    EVENT_EXP_CTR = 1
    EVENT_PREDICTION_ACCURACY = 2
    EVENT_SIGNIFICANCE_LEVEL_DATA = 3
    EVENT_ITERATION_COMPLETE = 4
    EVENT_DEBUG = 5

    def __init__(self, sim_id, results_queue, user_ftrs, p_random_select, p_random_learn, restart_queue, kill_queue):
        self.sim_id = sim_id
        user_type = Cheater if p_random_select < 0 else User
        self.user = user_type(user_ftrs)
        self.p_random_select = p_random_select
        self.p_random_learn = p_random_learn
        self.restart_queue = restart_queue
        self.results_queue = results_queue
        self.kill_queue = kill_queue

    def is_not_killed(self):
        return self.kill_queue.empty()

    def run(self):
        self.results_queue.put(
            (self.__class__.EVENT_DEBUG, self.sim_id, "Started!"))
        i = 0
        while self.is_not_killed():
            if self.restart_queue.get() == False:
                break
            i += 1
            ad = self.user.select_ad(p_random=self.p_random_select)
            did_click = self.user.interact_with(*ad)
            if self.is_not_killed():
                self.results_queue.put(
                    (self.__class__.EVENT_INTERACTION, self.sim_id, did_click))
                self.results_queue.put(
                    (self.__class__.EVENT_EXP_CTR, self.sim_id, self.user.sum_selected_ctrs/i))
                self.user.remember(*ad, did_click)
                self.user.learn(p_random=self.p_random_learn)
                self.results_queue.put((
                    self.__class__.EVENT_PREDICTION_ACCURACY,
                    self.sim_id,
                    self.user.evaluate_prediction_accuracy_for_available_ads(),
                    self.user.evaluate_prediction_accuracy_for_ads_in_memory()
                ))
                self.results_queue.put((
                    self.__class__.EVENT_SIGNIFICANCE_LEVEL_DATA,
                    self.sim_id,
                    self.user.random_mean_ctrs,
                    self.user.cheater_ctrs,
                    np.sum(self.user.interaction_history)
                ))
                self.results_queue.put(
                    (self.__class__.EVENT_ITERATION_COMPLETE, self.sim_id))


def run_sim(*args):
    Simulation(*args).run()


def run_n_sims(new_data, args_list):
    set_data(new_data)
    for args in args_list:
        threading.Thread(target=run_sim, args=args).start()
