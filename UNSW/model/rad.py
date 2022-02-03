import pandas as pd
import torch
from UNSW.model import a2c, protonet
import torch.optim as optim
import numpy as np
from torch.distributions import Categorical
from collections import namedtuple
from tqdm import tqdm
import math
from numpy.linalg import norm


def generate_samples(df_unseen_pseudo, num_samples=100):
    df_unseen = df_unseen_pseudo.copy()
    n_clusters = set(df_unseen['y_pred'].values)
    for ind, val in enumerate(n_clusters):
        if ind == 0:
            if val == 0:
                if len(df_unseen.loc[df_unseen['y_pred'] == val]) >= num_samples*5:
                    df_samples = df_unseen.loc[df_unseen['y_pred'] == val].sample(num_samples*5, random_state=42)
                else:
                    df_samples = df_unseen.loc[df_unseen['y_pred'] == val]
            else:
                if len(df_unseen.loc[df_unseen['y_pred'] == val]) >= num_samples:
                    df_samples = df_unseen.loc[df_unseen['y_pred'] == val].sample(num_samples, random_state=42)
                else:
                    df_samples = df_unseen.loc[df_unseen['y_pred'] == val]
        else:
            if val == 0:
                if len(df_unseen.loc[df_unseen['y_pred'] == val]) >= num_samples*5:
                    df_samples = pd.concat(
                        [df_samples, df_unseen.loc[df_unseen['y_pred'] == val].sample(num_samples*5, random_state=42)])
                else:
                    df_samples = pd.concat([df_samples, df_unseen.loc[df_unseen['y_pred'] == val]])
            else:
                if len(df_unseen.loc[df_unseen['y_pred'] == val]) >= num_samples:
                    df_samples = pd.concat(
                        [df_samples, df_unseen.loc[df_unseen['y_pred'] == val].sample(num_samples, random_state=42)])
                else:
                    df_samples = pd.concat([df_samples, df_unseen.loc[df_unseen['y_pred'] == val]])
    df_unseen.drop(df_samples.index.values, inplace=True)
    df_unseen.reset_index(drop=True, inplace=True)
    df_samples.reset_index(drop=True, inplace=True)
    return df_unseen, df_samples


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


def selected_combination(df_seen, df_unseen, df_samples, actions):
    df_samples['action'] = actions
    df_samples1 = df_samples.loc[df_samples['action'] == 1].copy()
    df_samples1.drop(columns=['action'], inplace=True)
    df_samples1.reset_index(drop=True, inplace=True)
    df_seen = pd.concat([df_seen, df_samples1]).reset_index(drop=True)
    df_samples0 = df_samples.loc[df_samples['action'] == 0].copy()
    df_samples0.drop(columns=['action'], inplace=True)
    df_samples0.reset_index(drop=True, inplace=True)
    df_unseen = pd.concat([df_unseen, df_samples0]).reset_index(drop=True)
    return df_seen, df_unseen


class RAD(object):
    def __init__(self, options):
        super(RAD, self).__init__()
        self.device = options['device']
        self.policy = a2c.Policy(options).to(self.device)
        self.optim = optim.Adam(self.policy.parameters())
        self.eps = np.finfo(np.float32).eps.item()
        self.r_prev = None
        self.r_new = None
        self.max_episode = options['max_episode']
        self.max_iterators = options['max_iterators']
        self.options = options
        self.lamb = options['lambda']
        self.ad_alpha = options['r_ad_alpha']
        self.cl_alpha = options['r_cl_alpha']
        self.num_samples = options['num_samples']

        self.saved_actions = []
        self.rewards = []


    def select_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        probs, state_value = self.policy(state)

        # create a categorical distribution over the list of probabilities of actions
        m = Categorical(probs)

        # and sample an action using the distribution
        action = m.sample()

        # save to action buffer
        for i in range(len(action)):
            self.saved_actions.append(SavedAction(m.log_prob(action[i]), state_value[i]))

        action = action.detach().cpu().numpy()
        return action

    def finish_episode(self, r_true):

        saved_actions = self.saved_actions
        policy_losses = []  # list to save actor (policy) loss
        value_losses = []  # list to save critic (value) loss
        returns = []  # list to save the true values

        returns = [(r_true / len(saved_actions)) for _ in range(len(saved_actions))]
        returns = torch.tensor(returns)

        for (log_prob, value), R in zip(saved_actions, returns):
            advantage = R - value.item()

            # calculate actor (policy) loss
            policy_losses.append(-log_prob * advantage)

            # calculate critic (value) loss using L1 smooth loss
            value_losses.append(abs(value - torch.tensor([R]).to(self.device)))
        del self.saved_actions[:]
        return torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    def finish_iterator(self, loss_lst):
        # reset gradients
        self.optim.zero_grad()

        # sum up all the values of policy_losses and value_losses
        loss = sum(loss_lst)

        # perform backprop
        loss.backward()
        self.optim.step()

    def train_rad(self, df_seen, df_unseen, df_sup, test_x=None, test_y=None):
        prototrainer = protonet.ProtoTrainer(self.options)
        for i_iterator in range(self.max_iterators):
            if i_iterator == 0:
                df_seen, df_seen_eval, df_unseen_pseudo, seen_f1_ad, seen_f1 = prototrainer.training_first(df_seen, df_unseen, df_sup, test_x, test_y)
                prototrainer.training_baseline(df_seen, df_seen_eval, df_unseen, test_x, test_y)
            else:
                n_min = max(0, math.floor(min(df_seen.loc[df_seen['y_pred'] > 0].groupby('y_pred').count()['y_true'])*0.6)//2)
                if n_min >= self.options['n_support']:
                    self.options['n_support'] = n_min
                    self.options['n_query'] = n_min
                    print(self.options['n_support'], self.options['n_query'])
                    prototrainer.n_query = self.options['n_query']
                    prototrainer.n_support = self.options['n_support']
                df_seen, df_seen_eval, df_unseen_pseudo, seen_f1_ad, seen_f1 = prototrainer.training_before(df_seen, df_unseen)
            r_before = seen_f1_ad*self.ad_alpha + seen_f1*self.cl_alpha
            max_reward = r_before
            max_seen = df_seen.copy()
            max_unseen = df_unseen_pseudo.copy()
            # loss_lst = []
            for _ in tqdm(range(self.max_episode), desc="iterator {:d} train".format(i_iterator + 1)):
                loss_lst = []
                df_unseen, df_samples = generate_samples(df_unseen_pseudo, self.options['num_samples'])
                df_samples_emb = prototrainer.in_embedding(df_samples, i_iterator)
                actions = self.select_action(df_samples_emb.values)
                df_seen_episode, df_unseen_episode = selected_combination(df_seen, df_unseen, df_samples, actions)
                seen_f1_ad, seen_f1 = prototrainer.training_after(df_seen_episode, df_seen_eval)
                r_episode = seen_f1_ad*self.ad_alpha + seen_f1*self.cl_alpha
                loss = self.finish_episode(norm(r_episode - r_before))
                loss_lst.append(loss)
                if r_episode > max_reward:
                    prototrainer.save_best_model()
                    max_seen = df_seen_episode.copy()
                    max_unseen = df_unseen_episode.copy()
                    max_reward = r_episode
                self.finish_iterator(loss_lst)
            # self.finish_iterator(loss_lst)
            print(f'prev reward {r_before} best reward {max_reward}')
            if max_reward > r_before:
                prototrainer.update_best_model()
                df_seen = max_seen.copy()
                df_unseen = max_unseen.copy()
            df_seen = pd.concat([df_seen, df_seen_eval])
            df_seen.reset_index(drop=True, inplace=True)
            if (test_x is not None) and (min(df_seen.groupby(['y_pred']).count()['y_true'].values) >= self.options['n_min_size']) and (len(df_seen.groupby(['y_pred']).count()) == self.options['n_ways']):
                prototrainer.final_training_testing(df_seen, test_x, test_y)
        prototrainer.final_training_testing(df_seen, test_x, test_y)
        df_unseen_pseudo = df_unseen
        return df_seen, df_unseen_pseudo
