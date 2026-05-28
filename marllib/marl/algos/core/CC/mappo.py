# MIT License

# Copyright (c) 2023 Replicable-MARL

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy, KLCoeffMixin
from ray.rllib.algorithms.ppo.ppo import PPO as PPOTrainer, DEFAULT_CONFIG as PPO_CONFIG
from ray.rllib.models.action_dist import ActionDistribution
import gym
from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.policy.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.policy.torch_mixins import EntropyCoeffSchedule, \
    LearningRateSchedule
from ray.rllib.utils.typing import TensorType, TrainerConfigDict
from marllib.marl.algos.utils.centralized_critic import CentralizedValueMixin, centralized_critic_postprocessing
from marllib.marl.algos.core import setup_torch_mixins

#############
### MAPPO ###
#############

def central_critic_ppo_loss(policy: Policy, model: ModelV2,
                            dist_class: ActionDistribution,
                            train_batch: SampleBatch) -> TensorType:
    """Constructs the loss for Centralized PPO Objective.
    Args:
        policy (Policy): The Policy to calculate the loss for.
        model (ModelV2): The Model to calculate the loss for.
        dist_class (Type[ActionDistribution]: The action distr. class.
        train_batch (SampleBatch): The training data.

    Returns:
        Union[TensorType, List[TensorType]]: A single loss tensor or a list
            of loss tensors.
    """
    CentralizedValueMixin.__init__(policy)

    vf_saved = model.value_function
    opp_action_in_cc = policy.config["model"]["custom_model_config"]["opp_action_in_cc"]
    model.value_function = lambda: policy.model.central_value_function(train_batch["state"],
                                                                       train_batch[
                                                                           "opponent_actions"] if opp_action_in_cc else None)

    policy._central_value_out = model.value_function()
    # Call PPOTorchPolicy.loss directly (replaces removed ppo_surrogate_loss)
    loss = PPOTorchPolicy.loss(policy, model, dist_class, train_batch)

    model.value_function = vf_saved

    return loss


class MAPPOTorchPolicy(CentralizedValueMixin, PPOTorchPolicy):
    """MAPPO policy: PPO with centralized critic.

    PPOTorchPolicy in Ray 2.x already inherits LearningRateSchedule,
    EntropyCoeffSchedule, KLCoeffMixin, and ValueNetworkMixin, so we only
    add CentralizedValueMixin on top.
    """

    def __init__(self, observation_space, action_space, config):
        PPOTorchPolicy.__init__(self, observation_space, action_space, config)
        CentralizedValueMixin.__init__(self)

    def loss(self, model, dist_class, train_batch):
        return central_critic_ppo_loss(self, model, dist_class, train_batch)

    def postprocess_trajectory(self, sample_batch, other_agent_batches=None,
                               episode=None):
        return centralized_critic_postprocessing(
            self, sample_batch, other_agent_batches, episode)


def get_policy_class_mappo(config_):
    if config_["framework"] == "torch":
        return MAPPOTorchPolicy


class MAPPOTrainer(PPOTrainer):
    """MAPPO trainer: PPO with centralized critic."""

    @classmethod
    def get_default_policy_class(cls, config):
        return get_policy_class_mappo(config)
