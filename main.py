import torch
import msvcrt

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from agent import Agent

from nes_py.wrappers import JoypadSpace
from wrappers import apply_wrappers

import os

from utils import *

from file_management import File_Management

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

file_managment = File_Management()


def print_menu():
    print("0: print this menu")
    print("1: load savepoint")
    print("2: start new training")
    print("3: load best trainings")


def load_savepoint():
    savepoint_filpath = get_savepoint_file()
    if savepoint_filpath is None:
        print("Unable to load savepoint")
        return


def get_savepoint_file():
    files = file_managment.get_savepoint_files()
    if not files:
        print("No save points found")
        return None

    savepoint_selected = False

    while not savepoint_selected:
        print(f"{len(files)} savepoints found. Select which savepoint to load...")
        print("0 - Cancel loading savepoint")
        for i in range(0, len(files)):
            print(f"{i + 1} - {files[i]}")

        try:
            selected_savepoint = int(input("Enter Command: "))
            if selected_savepoint == 0:
                print("Canceling loading savepoint")
                return None
            elif 1 <= selected_savepoint <= len(files):
                savepoint = files[selected_savepoint - 1]
                print(f"Loading savepoint {savepoint}")
                return os.path.join(file_managment.savepoints_filepath, savepoint)
            else:
                print("Invalid index selected")
        except ValueError:
            print("Invalid input, Please enter an int")


def train(training_params):
    model_path = os.path.join("models", get_current_date_time_string())
    os.makedirs(model_path, exist_ok=True)

    """
    checkpoint_path = file_managment.get_first_file(
        file_managment.savepoints_filepath
    )  # "checkpoint.pt"
    resume = False if checkpoint_path == None else os.path.exists(checkpoint_path)
    """

    ENV_NAME = "SuperMarioBros-1-1-v0"
    SHOULD_TRAIN = True
    DISPLAY = True
    CKPT_SAVE_INTERVAL = 1000
    NUM_OF_EPISODES = 50_000

    env = gym_super_mario_bros.make(
        ENV_NAME,
        render_mode="human" if DISPLAY else "rgb",
        apply_api_compatibility=True,
    )
    env = JoypadSpace(env, RIGHT_ONLY)

    env = apply_wrappers(env)

    agent = Agent(
        input_dims=env.observation_space.shape, num_actions=env.action_space.n
    )

    if not SHOULD_TRAIN:
        folder_name = ""
        ckpt_name = ""
        agent.load_model(os.path.join("models", folder_name, ckpt_name))
        agent.epsilon = 0.2
        agent.eps_min = 0.0
        agent.eps_decay = 0.0

    env.reset()
    next_state, reward, done, trunc, info = env.step(action=0)

    """
    if resume:
        start_episode = agent.load_checkpoint(checkpoint_path)
        print(f"resuming at episode {agent.training_iteration}")
    """
    start_episode = agent.training_iteration

    for i in range(start_episode, NUM_OF_EPISODES):
        agent.training_iteration = i

        if msvcrt.kbhit():
            key = msvcrt.getch().decode("utf-8").lower()
            if key == "q":
                print("Quiting training loop")
                agent.save_checkpoint(file_managment.get_savepoint_path())
                break

        print("Episode:", i)
        done = False
        state, _ = env.reset()
        total_reward = 0
        while not done:
            a = agent.choose_action(state)
            new_state, reward, done, truncated, info = env.step(a)
            total_reward += reward

            # if DISPLAY:
            #    time.sleep(1)

            if SHOULD_TRAIN:
                agent.store_in_memory(state, a, reward, new_state, done)
                agent.learn()

            state = new_state

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
            agent.save_model(
                os.path.join(model_path, "model_" + str(i + 1) + "_iter.pt")
            )

        print("Total reward:", total_reward)

    env.close()


def start_training():
    training_params = []
    checkpoint

    train(training_params)


quit_program = False
while not quit_program:
    command = input("Enter Command: ").strip()

    if command == "0":
        print_menu()

    elif command == "1":
        load_savepoint()

    elif command == "2":
        start_training()
