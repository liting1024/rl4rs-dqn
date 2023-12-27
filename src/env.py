# coding:utf-8
import numpy as np
import torch
import pdb


class Env:
    def __init__(self, value, K=3):
        self.K = K - 1
        self.value = np.asarray(value)

    def done(self, obs):
        return obs[3] is not None and obs[3].size(1) == self.K

    def __recall(self, s, t):
        """
        计算s和t的召回率
        """
        return sum(i in t for i in s) / len(t)

    def __reward(self, s, t):
        """
        奖赏=召回率x值
        """
        return self.__recall(s, t) * self.value[s].sum()

    def new_obs(self, batch_obs, batch_actions):
        """
        执行动作后生成新的观测
        接收当前的 batch_obs 和执行的动作 batch_actions，然后将动作添加到已暴露的物品中，生成新的批量观测 batch_new_obs。
        如果当前已经观测到了所有物品，返回 None，表示回合结束。否则，返回包含新观测的元组。
        """
        batch_users, batch_click_items, batch_click_mask, \
            batch_exposed_items, batch_exposed_mask = batch_obs

        batch_new_exposed_items = torch.cat(
            [batch_exposed_items, batch_actions.unsqueeze(1)], dim=1
        ) if batch_exposed_items is not None else batch_actions.unsqueeze(1)

        _add_mask = torch.tensor([[True]]).expand(batch_users.size(0), -1)
        batch_new_exposed_mask = torch.cat(
            [batch_exposed_mask, _add_mask], dim=1
        ) if batch_exposed_mask is not None else _add_mask

        batch_new_obs = (batch_users, batch_click_items, batch_click_mask,
                         batch_new_exposed_items, batch_new_exposed_mask)
        return batch_new_obs

    def step(self, batch_obs, batch_actions, batch_target_bundles, time):
        batch_rews = torch.tensor([self.__reward(action, bundle) \
                                   for action, bundle in zip(batch_actions, batch_target_bundles[:, time])],
                                  dtype=torch.float32)
        batch_users, batch_click_items, batch_click_mask, batch_exposed_items, batch_exposed_mask = batch_obs
        done = batch_exposed_mask is not None and batch_exposed_mask[0].sum() == self.K
        if done:
            batch_new_obs = [None] * batch_users.size(0)
        else:
            batch_new_obs = self.new_obs(batch_obs, batch_actions)

        return batch_new_obs, batch_rews, torch.tensor([done] * batch_actions.size(0))

    # def step(self, batch_obs, batch_actions):
    #    batch_rews = torch.tensor([self.prices[action].sum() for action in batch_actions])
    #    batch_users, batch_click_items, batch_click_mask, \
    #            batch_exposed_items, batch_exposed_mask = batch_obs
    #    #batch_done = [exposed_mask.sum() == self.K for exposed_mask in batch_exposed_mask])
    #    done = batch_exposed_mask is not None and batch_exposed_mask[0].sum() == self.K
    #    if done:
    #        batch_rews += torch.tensor([(self.prices[exposed_items.view(-1)]).sum() \
    #                                    for exposed_items in batch_obs[3]])
    #        batch_new_obs = None
    #    else:
    #        batch_new_exposed_items = torch.cat(
    #            [batch_exposed_items, batch_actions.unsqueeze(1)], dim=1
    #        ) if batch_exposed_items is not None else batch_actions.unsqueeze(1)
    #        _add_mask = torch.tensor([[True]]).expand(batch_users.size(0), -1)
    #        batch_new_exposed_mask = torch.cat(
    #            [batch_exposed_mask, _add_mask], dim=1
    #        ) if batch_exposed_mask is not None else _add_mask
    #        batch_new_obs = (batch_users, batch_click_items, batch_click_mask, \
    #                         batch_new_exposed_items, batch_new_exposed_mask)
    #    return batch_new_obs, batch_rews, done
