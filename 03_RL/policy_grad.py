import torch
from torch.autograd import Variable
import torch.nn.functional as F

def compute_losses(R, ep_rewards, ep_values, ep_action_log_probs, masks, gamma, tau, algo):
    """ Compute the policy and value func losses given the terminal reward, the episode rewards,
    values, and action log probabilities """
    policy_loss = 0.0
    value_loss = 0.0

    # TODO: Use the available parameters to compute the policy gradient loss and the value function loss.
    values = torch.stack(ep_values).squeeze(-1).squeeze(-1)  # [T]
    log_probs = torch.stack(ep_action_log_probs).view(-1)  # [T]
    device = values.device

    rewards = torch.as_tensor(ep_rewards, dtype=torch.float32, device=device)  # [T]
    masks_t = torch.as_tensor(masks[:-1], dtype=torch.float32, device=device)  # length T
    next_value = torch.as_tensor(R, dtype=torch.float32, device=device).view(())  # scalar

    gae = torch.tensor(0.0, device=device)
    advs, rets = [], []
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value * masks_t[t] - values[t]
        gae = delta + gamma * tau * masks_t[t] * gae
        advs.append(gae)
        rets.append(values[t] + gae)
        next_value = values[t].detach()

    advantages = torch.stack(advs[::-1])  # [T]
    returns = torch.stack(rets[::-1])  # [T]

    if advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

    policy_loss = -(log_probs * advantages.detach()).sum()
    value_loss = 0.5 * F.mse_loss(values, returns.detach(), reduction="sum")  # shapes now match
    return policy_loss, value_loss
    # raise NotImplementedError("Compute the policy and value function loss.")
