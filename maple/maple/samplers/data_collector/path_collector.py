import random
from collections import deque, OrderedDict
from functools import partial

import numpy as np

from maple.core.eval_util import create_stats_ordered_dict
from maple.samplers.data_collector.base import PathCollector
from maple.samplers.rollout_functions import rollout, rollout_evo


class MdpPathCollector(PathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,
            render=False,
            render_kwargs=None,
            rollout_fn=rollout,
            save_env_in_snapshot=True,
            rollout_fn_kwargs=None,
            LLM_actions=None,
            LLM_weight=0,
    ):
        if render_kwargs is None:
            render_kwargs = {}
        self._env = env
        self.LLM_weight = LLM_weight
        self._policy = policy
        self._max_num_epoch_paths_saved = max_num_epoch_paths_saved
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)
        self._render = render
        self._render_kwargs = render_kwargs
        self._rollout_fn = rollout_fn
        if rollout_fn_kwargs is None:
            rollout_fn_kwargs = {}
        if LLM_actions is None:
            LLM_actions = {}
        self._rollout_fn_kwargs = rollout_fn_kwargs
        self.LLM_actions = LLM_actions
        self._num_steps_total = 0
        self._num_paths_total = 0

        self._num_actions_total = 0

        self._save_env_in_snapshot = save_env_in_snapshot

    def collect_new_paths(
            self,
            max_path_length,
            num_steps,
            discard_incomplete_paths,
    ):
        paths = []
        num_steps_collected = 0
        num_actions_collected = 0
        while num_steps_collected < num_steps:
            max_path_length_this_loop = min(  # Do not go over num_steps
                max_path_length,
                num_steps - num_steps_collected,
            )
            if discard_incomplete_paths and (max_path_length_this_loop < max_path_length):
                break
            if random.random() < self.LLM_weight:
                LLM_mode = True
                self.LLM_weight = self.LLM_weight*0.99999
                print(self.LLM_weight)
            else:
                LLM_mode = False
            path = self._rollout_fn(
                self._env,
                self._policy,
                max_path_length=max_path_length_this_loop,
                render=self._render,
                render_kwargs=self._render_kwargs,
                LLM_actions=self.LLM_actions,
                LLM_mode=LLM_mode,
                **self._rollout_fn_kwargs
            )

            num_steps_collected += path['path_length']
            num_actions_collected += path['path_length_actions']
            paths.append(path)
        self._num_paths_total += len(paths)
        self._num_steps_total += num_steps_collected
        self._num_actions_total += num_actions_collected
        self._epoch_paths.extend(paths)
        return paths

    def get_epoch_paths(self):
        return self._epoch_paths

    def end_epoch(self, epoch):
        self._epoch_paths = deque(maxlen=self._max_num_epoch_paths_saved)

    def get_diagnostics(self):
        # path_lens = [len(path['actions']) for path in self._epoch_paths]
        path_lens = [path['path_length'] for path in self._epoch_paths]
        stats = OrderedDict([
            ('num steps total', self._num_steps_total),
            ('num paths total', self._num_paths_total),
            ('num actions total', self._num_actions_total),
        ])
        stats.update(create_stats_ordered_dict(
            "path length",
            path_lens,
            always_show_all_stats=True,
        ))
        return stats

    def get_snapshot(self):
        snapshot_dict = dict(
            policy=self._policy,
        )
        if self._save_env_in_snapshot:
            snapshot_dict['env'] = self._env
        return snapshot_dict


class GoalConditionedPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            desired_goal_key='desired_goal',
            goal_sampling_mode=None,
            **kwargs
    ):
        def obs_processor(o):
            return np.hstack((o[observation_key], o[desired_goal_key]))

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key
        self._desired_goal_key = desired_goal_key
        self._goal_sampling_mode = goal_sampling_mode

    def collect_new_paths(self, *args, **kwargs):
        self._env.goal_sampling_mode = self._goal_sampling_mode
        return super().collect_new_paths(*args, **kwargs)

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
            desired_goal_key=self._desired_goal_key,
        )
        return snapshot


class ObsDictPathCollector(MdpPathCollector):
    def __init__(
            self,
            *args,
            observation_key='observation',
            **kwargs
    ):
        def obs_processor(obs):
            return obs[observation_key]

        rollout_fn = partial(
            rollout,
            preprocess_obs_for_policy_fn=obs_processor,
        )
        super().__init__(*args, rollout_fn=rollout_fn, **kwargs)
        self._observation_key = observation_key

    def get_snapshot(self):
        snapshot = super().get_snapshot()
        snapshot.update(
            observation_key=self._observation_key,
        )
        return snapshot


class VAEWrappedEnvPathCollector(GoalConditionedPathCollector):
    def __init__(
            self,
            env,
            policy,
            decode_goals=False,
            **kwargs
    ):
        """Expects env is VAEWrappedEnv"""
        super().__init__(env, policy, **kwargs)
        self._decode_goals = decode_goals

    def collect_new_paths(self, *args, **kwargs):
        self._env.decode_goals = self._decode_goals
        return super().collect_new_paths(*args, **kwargs)


class EvoPathCollector(MdpPathCollector):
    def __init__(
            self,
            env,
            policy,
            max_num_epoch_paths_saved=None,  # None 表示无上限
            render=False,
            render_kwargs=None,
            rollout_fn=rollout_evo,
            save_env_in_snapshot=True,
            rollout_fn_kwargs=None,
            population=None,
    ):
        super().__init__(
            env,
            policy,
            max_num_epoch_paths_saved=max_num_epoch_paths_saved,  # None 表示无上限
            render=render,
            render_kwargs=render_kwargs,
            rollout_fn=rollout_fn,
            save_env_in_snapshot=save_env_in_snapshot,
            rollout_fn_kwargs=rollout_fn_kwargs,
        )
        self.population = population

    def collect_new_paths(
            self,
            max_path_length,  # 每次完整交互执行的最大动作基元个数
            num_steps,  # 需要执行的总的动作基元个数
            discard_incomplete_paths,
            success_memory=True,
    ):
        paths = []
        # 执行的步骤
        num_steps_collected = 0
        # 执行的动作
        num_actions_collected = 0
        id = 0
        if success_memory:
            actions_buffer, fitness_id = self.population.get_batch_from_success_memory()
        else:
            actions_buffer, fitness_id = self.population.get_batch_from_var_memory()

        for actions in actions_buffer:
            fitness = 0
            for i in range(1):
                path = self._rollout_fn(
                    self._env,
                    self._policy,  # agent--->PDMDPPolicy
                    render=self._render,
                    render_kwargs=self._render_kwargs,
                    evo=True,
                    actions_population=actions,
                    **self._rollout_fn_kwargs
                )
                # 下面两个相同，执行的步骤和执行的动作基元个数一样
                num_steps_collected += path['path_length']
                num_actions_collected += path['path_length_actions']
                paths.append(path)
                fitness = fitness + np.sum([x for x in path['rewards']])
            if success_memory:
                self.population.successful_memory_fitness[fitness_id[id]] = (fitness) * 0.1 + \
                                                                            self.population.successful_memory_fitness[
                                                                                fitness_id[id]] * 0.9
            else:
                memory_id = fitness_id[id]
                self.population.var_memory_fitness[memory_id] = (self.population.var_memory_fitness[memory_id] * (
                            self.population.ucb_var_chosen[memory_id] - 1) + fitness) / self.population.ucb_var_chosen[
                                                                    memory_id]


            self._num_paths_total += len(paths)
            self._num_steps_total += num_steps_collected
            self._num_actions_total += num_actions_collected
            self._epoch_paths.extend(paths)
            id = id + 1
        return paths
