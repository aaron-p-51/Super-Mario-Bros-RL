import torch
import msvcrt

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from Controller.controller_visualizer import ControllerVisualizer

from agent import Agent

from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers

import os

from utils import *

# Create model path for checkpoints
model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)

save_path = os.path.join("models", "savepoints")
os.makedirs(save_path, exist_ok=True)

# visualizer = ControllerVisualizer(button_names=["left", "right", "A", "B"])

############################ Configuration #########################
SAVEPOINT_FILE = "2025-05-08-16_36_54_.pt"

ENV_NAME = "SuperMarioBros-1-1-v0"
FRAME_DELAY = 0
SHOULD_TRAIN = True
DISPLAY = True
CKPT_SAVE_INTERVAL = 1000
NUM_OF_EPISODES = 50_000
STUCK_LIMIT = 60


if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")


env = gym_super_mario_bros.make(
    ENV_NAME, render_mode="human" if DISPLAY else "rgb", apply_api_compatibility=True
)
env = JoypadSpace(env, SIMPLE_MOVEMENT)

env = apply_wrappers(env)

agent = Agent(input_dims=env.observation_space.shape, num_actions=env.action_space.n)

if not SHOULD_TRAIN:
    folder_name = ""
    ckpt_name = ""
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.2
    agent.eps_min = 0.0
    agent.eps_decay = 0.0

env.reset()
next_state, reward, done, trunc, info = env.step(action=0)

start_episode = agent.training_iteration

savepoint_path = os.path.join("models", "savepoints", SAVEPOINT_FILE)
if SAVEPOINT_FILE != "" and os.path.exists(savepoint_path):
    agent.load_savepoint(savepoint_path)
    start_episode = agent.training_iteration

for i in range(start_episode, NUM_OF_EPISODES):
    print("Episode:", i)
    done = False
    state, _ = env.reset()
    total_reward = 0
    agent.training_iteration = i
    stuck_counter = 0
    prev_x_pos = 0

    if msvcrt.kbhit():
        key = msvcrt.getch().decode("utf-8").lower()
        if key == "q":
            print("Quiting training loop")
            new_save_point = os.path.join(
                save_path, get_current_date_time_string() + "_.pt"
            )
            print(f"Savepoint created at: {new_save_point}")
            agent.save_savepoint(new_save_point)
            break

    while not done:
        a = agent.choose_action(state)
        buttons = SIMPLE_MOVEMENT[a]
        # visualizer.update(buttons)
        new_state, reward, done, truncated, info = env.step(a)
        total_reward += reward

        if DISPLAY and FRAME_DELAY > 0:
            time.sleep(FRAME_DELAY)

        if SHOULD_TRAIN:
            agent.store_in_memory(state, a, reward, new_state, done)
            agent.learn()

        state = new_state

        if info["x_pos"] > prev_x_pos:
            prev_x_pos = info["x_pos"]
            stuck_counter = 0
        else:
            stuck_counter += 1

        # Check if Mario is stuck for too long
        if stuck_counter >= STUCK_LIMIT:
            print(f"Mario is stuck. Ending episode {i} early")
            done = True

    print(
        "Total reward:",
        total_reward,
        "Epsilon:",
        agent.epsilon,
        "Size of replay buffer:",
        len(agent.replay_buffer),
        "Learn step counter:",
        agent.learn_step_counter,
    )

    if SHOULD_TRAIN and (i + 1) % CKPT_SAVE_INTERVAL == 0:
        agent.save_model(os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt"))

    print("Total reward:", total_reward)

env.close()
