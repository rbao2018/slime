import torch
from slime.utils.types import Sample

__all__ = ["is_reward_zero_std", "more_than_half_correct", "is_all_correct", "is_all_false", "is_not_valid_group"]

def is_reward_zero_std(args, samples: list[Sample], **kwargs):
    if len(samples) != args.n_samples_per_prompt:
        return True
    rewards = [sample.get_reward_value(args) for sample in samples]
    
    # 如果标准差小于阈值，认为所有值相同
    std_val = torch.tensor(rewards, dtype=torch.float).std()
    return std_val <= 1e-7  # 使用小阈值而不是严格的0比较

def more_than_half_correct(args, samples: list[Sample], **kwargs):
    """
    Return True if strictly less than half of the samples have a positive reward.
    """
    if len(samples) != args.n_samples_per_prompt:
        return True
    rewards = [sample.get_reward_value(args) for sample in samples]
    positive = sum(r > 0 for r in rewards)
    return positive >= len(samples) // 2


def is_all_correct(args, samples: list[Sample], **kwargs):
    if len(samples) != args.n_samples_per_prompt:
        return True
    rewards = [sample.get_reward_value(args) for sample in samples]
    return all(r > 1 for r in rewards)


def is_all_false(args, samples: list[Sample], **kwargs):
    if len(samples) != args.n_samples_per_prompt:
        return True
    rewards = [sample.get_reward_value(args) for sample in samples]
    return all(r < 1 for r in rewards)

def is_not_valid_group(args, samples: list[Sample], **kwargs):
    if len(samples) != args.n_samples_per_prompt:
        return True
    return False