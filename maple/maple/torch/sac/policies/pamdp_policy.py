from maple.torch.sac.policies.base import TorchStochasticPolicy
from maple.torch.core import PyTorchModule
from maple.torch.networks import Mlp

from maple.torch.distributions import (
    Softmax, TanhNormal,
    ConcatDistribution,
    HierarchicalDistribution,
)

import torch
from torch import nn as nn

LOGITS_SCALE = 10
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class PAMDPPolicy(PyTorchModule, TorchStochasticPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim_s,
            action_dim_p,
            one_hot_s,
    ):
        super().__init__()

        task_policy = CategoricalPolicy(
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            action_dim=action_dim_s,
            one_hot=one_hot_s,
            prefix='task',
        )

        param_policy = ParallelHybridPolicy(
            obs_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            num_networks=action_dim_s,
            action_dim_c=action_dim_p,
            prefix_c='param',
        )

        self.policy = HierarchicalPolicy(
            task_policy, param_policy,
            policy1_obs_dim=obs_dim
        )

        self.obs_dim = obs_dim
        self.action_dim_s = action_dim_s
        self.action_dim_p = action_dim_p

        self.one_hot_s = one_hot_s

    def forward(self, obs):
        return self.policy(obs)

class ConcatPolicy(PyTorchModule, TorchStochasticPolicy):
    def __init__(
            self,
            policy1,
            policy2,
            policy1_obs_dim
    ):
        super().__init__()

        self.policy1 = policy1
        self.policy2 = policy2

        self.policy1_obs_dim = policy1_obs_dim

    def forward(self, obs):
        return ConcatDistribution(
            distr1=self.policy1(obs[:,-self.policy1_obs_dim:]),
            distr2=self.policy2(obs),
        )

class HierarchicalPolicy(PyTorchModule, TorchStochasticPolicy):
    def __init__(self, policy1, policy2, policy1_obs_dim):
        super().__init__()

        self.policy1 = policy1
        self.policy2 = policy2

        self.policy1_obs_dim = policy1_obs_dim

    def forward(self, obs):
        assert obs.dim() == 2

        def distr2_cond_fn(inputs):
            obs_for_p = obs
            if inputs.dim() == 3:
                tile_dim = inputs.shape[1]
                obs_for_p = obs.unsqueeze(1).repeat((1, tile_dim, 1))
            if isinstance(self.policy2, ParallelHybridPolicy):
                id = torch.argmax(inputs, dim=-1)
                return self.policy2(obs_for_p, id)
            else:
                return self.policy2(torch.cat([obs_for_p, inputs], dim=-1))

        return HierarchicalDistribution(
            distr1=self.policy1(obs[:,-self.policy1_obs_dim:]),
            distr2_cond_fn=distr2_cond_fn,
        )

class CategoricalPolicy(Mlp, TorchStochasticPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            prefix='',
            init_w=1e-3,
            one_hot=False,
            **kwargs
    ):
        super().__init__(
            hidden_sizes,
            input_size=obs_dim,
            output_size=action_dim,
            init_w=init_w,
            **kwargs
        )

        self.prefix = prefix
        self.one_hot = one_hot

    def forward(self, obs):
        h = obs
        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        logits = self.last_fc(h)
        logits = torch.clamp(logits, -LOGITS_SCALE, LOGITS_SCALE)
        return Softmax(logits, one_hot=self.one_hot, prefix=self.prefix)


class ParallelHybridPolicy(PyTorchModule, TorchStochasticPolicy):
    """
    Usage:

    ```
    policy = ParallelHybridPolicy(...)
    """

    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            num_networks,
            action_dim_c,
            prefix_c='',
            init_w=1e-3,
    ):
        super().__init__()

        self.log_std = None
        self.num_networks = num_networks

        mlp_list = []
        output_size = 2*action_dim_c
        for i in range(num_networks):
            mlp = Mlp(
                hidden_sizes,
                input_size=obs_dim,
                output_size=output_size,
                init_w=init_w,
            )
            mlp_list.append(mlp)
        self.mlp_list = nn.ModuleList(mlp_list)

        self.action_dim_c = action_dim_c
        self.prefix_c = prefix_c

    def forward(self, obs, id):
        if torch.numel(id) == 1:
            mean, std = self.get_mean_std(obs, id)
        else:
            input_dims = id.shape
            obs = obs.reshape((-1, obs.shape[-1]))
            id = id.reshape(-1)

            means = []
            stds = []
            for i in range(self.num_networks):
                mean, std = self.get_mean_std(obs, i)
                means.append(mean)
                stds.append(std)

            means = torch.stack(means, dim=1)
            stds = torch.stack(stds, dim=1)

            mean = means[torch.arange(obs.size(0)), id]
            std = stds[torch.arange(obs.size(0)), id]

            ### reshape to original input dims
            mean = mean.reshape((*input_dims, -1))
            std = std.reshape((*input_dims, -1))

        distr_c = TanhNormal(mean, std, prefix=self.prefix_c)
        return distr_c

    def get_mean_std(self, obs, id):
        c_dim = self.action_dim_c
        nn_output = self.mlp_list[id](obs)
        mean = nn_output[..., -2 * c_dim:-c_dim]
        log_std = nn_output[..., -c_dim:]

        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = torch.exp(log_std)

        return mean, std