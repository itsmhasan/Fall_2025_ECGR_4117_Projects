"""
Assume simple discrete action policy gradient
"""

import argparse
import sys
import os
from PIL import Image
import torch.nn.functional as F

import torch
import torch.optim as optim
from torch.autograd import Variable
import torchvision.transforms as T

import numpy as np

import models
import policy_grad #policy_grad_framework
from simulator.cartpole import CartPoleEnv

import matplotlib.pyplot as plt
from torchvision.transforms import InterpolationMode

# --- add after train_rewards/eval_rewards definitions ---
train_epoch_avg, eval_epoch_avg = [], []

parser = argparse.ArgumentParser(description='RL Pol Grad')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='learning rate (default: 1e-3)')
parser.add_argument('--batch-size', type=int, default=1000,
                    help='training batch size (default: 1000)')
parser.add_argument('--epochs', type=int, default=200,
                    help='epochs (default: 200)')
parser.add_argument('--episode-len', type=int, default=np.inf,
                    help='max episode length (default: np.inf)')
parser.add_argument('--gamma', type=float, default=0.99,
                    help='discount factor (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.95,
                    help='GAE parameter (default: 0.95)')
parser.add_argument('--algo', type=str, default='aac',
                    help='RL algorithm, options vpg, aac, npg (default: aac)')
parser.add_argument('--eval-model', type=str, default='',
                    help='path to model to eval (default: '')')
args = parser.parse_args()

env = CartPoleEnv()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Device:", device)

log_file = None
if args.eval_model == "":
    log_file = open('logs/state_pixels/model_'+args.algo+'.log', 'w')
train_rewards = []
eval_rewards = []
max_reward, avg_reward = 0, 0


# Composition of functions to take tensor and conver to network input.
resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.BICUBIC),
                    T.ToTensor()])

# This is based on the code from gym.
screen_width = 600

# This function and get_screen are adapted from
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# If the image is not cropped with the cart at the center, then learning is
# much more difficult.
def get_cart_location():
    world_width = env.x_threshold * 2
    scale = screen_width / world_width
    return int(env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

def get_screen():
    screen = env.render(mode='rgb_array').transpose(
        (2, 0, 1))  # transpose into torch order (CHW)
    # Strip off the top and bottom of the screen
    screen = screen[:, 160:320]
    view_width = 320
    cart_location = get_cart_location()
    if cart_location < view_width // 2:
        slice_range = slice(view_width)
    elif cart_location > (screen_width - view_width // 2):
        slice_range = slice(-view_width, None)
    else:
        slice_range = slice(cart_location - view_width // 2,
                            cart_location + view_width // 2)
    # Strip off the edges, so that we have a square image centered on a cart
    screen = screen[:, :, slice_range]
    # Convert to float, rescare, convert to torch tensor
    # (this doesn't require a copy)
    screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
    screen = torch.from_numpy(screen)
    # Resize, and add a batch dimension (BCHW)
    return resize(screen).unsqueeze(0).to(device)


def eval(model, avgn=5, render=False):
    model.eval()
    if render:
        env.reset()
        env.render()

    eval_reward = 0.0
    for i in range(avgn):
        done = False
        obs = env.reset()
        last_screen = get_screen()
        current_screen = get_screen()
        state = current_screen - last_screen

        while not done:
            if render:
                env.render()

            last_screen = current_screen
            current_screen = get_screen()
            state = current_screen - last_screen

            action_lin, value = model(state)

            action_probs = F.softmax(action_lin, dim=1)
            action = torch.max(action_probs, dim=1)[1]

            action = action.detach().cpu()  # ADD

            obs, reward, done, _ = env.step(action.data.numpy().squeeze())
            eval_reward += reward

    eval_reward /= avgn
    eval_rewards.append(eval_reward)
    eval_epoch_avg.append(float(eval_reward))  # NEW
    sys.stdout.write("\r\nEval reward: %d \r\n" % (eval_reward))
    sys.stdout.flush()

    return eval_reward


def train(model, opt):
    global max_reward, avg_reward
    model.train()

    traj_rewards = 0
    per_epoch_ep_returns = []  # NEW
    # hold action vars and probs
    ep_actions = []
    ep_action_log_probs = []
    ep_values = []
    ep_rewards = []
    masks = [1]

    done = False
    obs = env.reset()
    last_screen = get_screen()
    current_screen = get_screen()
    state = current_screen - last_screen

    step = 0
    while step < args.batch_size:
        last_screen = current_screen
        current_screen = get_screen()
        # Use the difference in images to retain the Markov property. Besides
        # this change all the training code is the same.
        state = current_screen - last_screen

        action_lin, value = model(state)

        action_probs = F.softmax(action_lin, dim=1)
        action_log_probs = F.log_softmax(action_lin, dim=1)
        action = action_probs.multinomial(1)

        action = action.detach().cpu()  # ADD

        action_log_prob = action_log_probs[0, action.data.numpy().squeeze()]
        obs, reward, done, _ = env.step(action.data.numpy().squeeze())

        ep_values.append(value)
        ep_rewards.append(reward)
        ep_actions.append(action)
        ep_action_log_probs.append(action_log_prob)
        masks.append(1 if not done else 0)

        traj_rewards += reward

        step += 1

        if done:
            # print ("Episode reward: ", traj_rewards)
            train_rewards.append(traj_rewards)
            per_epoch_ep_returns.append(traj_rewards)  # NEW
            if traj_rewards > max_reward:
                max_reward = traj_rewards
            avg_reward = sum(train_rewards[-10:]) / len(train_rewards[-10:])
            # print (max_reward, avg_reward)
            sys.stdout.write("Training: max reward: %d, window (10) average reward: %d \r" % (max_reward, avg_reward))
            sys.stdout.flush()

            traj_rewards = 0
            obs = env.reset()
            last_screen = current_screen
            current_screen = get_screen()
            state = current_screen - last_screen
            # if not on final step, reset done
            if step != args.batch_size - 1:
                done = False

    R = torch.zeros(1, 1)
    R = R.to(device)  # ADD
    if not done:
        # get last state value est
        last_screen = current_screen
        current_screen = get_screen()
        state = current_screen - last_screen
        action_lin, value = model(state)
        R = value.data

    # Standardize the rewards
    ep_rewards = np.array(ep_rewards)
    ep_rewards -= np.mean(ep_rewards)
    ep_rewards /= np.std(ep_rewards) + 1e-5
    ep_rewards = list(ep_rewards)

    policy_loss, value_loss = policy_grad.compute_losses(R, ep_rewards, ep_values, ep_action_log_probs, masks, args.gamma, args.tau, args.algo)

    opt.zero_grad()
    (value_loss+policy_loss).backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 20.0)
    opt.step()

    # store average training return for this epoch  # NEW
    if per_epoch_ep_returns:
        train_epoch_avg.append(float(np.mean(per_epoch_ep_returns)))
    else:
        train_epoch_avg.append(0.0)


def main():
    if args.eval_model != '':
        model = torch.load(args.eval_model)
        fname = os.path.basename(args.eval_model)
        print (fname[6:9], fname[10:-4])
        eval(model, render=True)
        return

    model = models.MLPPolicy().to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)

    e = 0
    while e < args.epochs:
        train(model, opt)

        if e % 1 == 0:
            avg_eval = eval(model)
            log_file.write(str(avg_eval)+'\n')
            log_file.flush()
        if e % 10 == 0:
            torch.save(model, 'saved_models/state_pixels/model_'+str(args.algo)+'_'+str(e)+'.pth')

        e += 1

    # make the required two-curve plot  # NEW
    os.makedirs("plots", exist_ok=True)
    plt.figure()
    plt.plot(train_epoch_avg, label="Train Avg Return")
    plt.plot(eval_epoch_avg, label="Eval Avg Return")
    plt.xlabel("Epoch");
    plt.ylabel("Average Return")
    plt.title("Image Policy Gradient: Training vs Evaluation Return (200 epochs)")
    plt.legend();
    plt.tight_layout()
    plt.savefig("plots/returns_pixels.png", dpi=150)
    print("\nSaved plot to plots/returns_pixels.png")


if __name__ == '__main__':
    main()
