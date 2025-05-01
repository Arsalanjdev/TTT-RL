from typing import Tuple, Dict
from collections import Counter, defaultdict

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from environment import TicTacToeEnv

Gamma = 0.9
TEST_EPISODE = 20

State  = Tuple[int,...]
Action = int
# RewardKey   = Tuple[State, Action, State]
# TransitionKey = Tuple[State, Action]

class TicTacToeAgent:
    def __init__(self):
        """
        RL agent that gets trained to learn how to play tic-tac-toe.
        """
        self.env = TicTacToeEnv()
        self.state, *_ = self.env.reset()
        self.rewards: Dict[Tuple[State, Action, State], int] = defaultdict(int)
        self.transitions: Dict[Tuple[State, Action], Counter] = defaultdict(Counter)
        self.values: Dict[State, float] = defaultdict(float)

    def _state_tuple(self, board: np.ndarray) -> State:
        return tuple(board.flatten())

    def play_n_random_steps(self, n: int):
        for _ in range(n):
            state = self._state_tuple(self.state)
            action = np.random.randint(0,9)
            new_state_unflatten, reward, done = self.env.step(action)
            new_state = self._state_tuple(new_state_unflatten)
            self.rewards[(state, action, new_state)] = reward
            self.transitions[(state, action)][new_state] += 1

            if done:
                self.state, *_ = self.env.reset()
            else:
                self.state = new_state_unflatten

    def calc_action_value(self, state: State, action: Action) -> float:
        trans_counts = self.transitions[(state, action)]
        total = sum(trans_counts.values())
        if total == 0:
            return 0.0
        value = 0.0
        for new_state, count in trans_counts.items():
            reward = self.rewards[(state, action, new_state)]
            value += (count/total) * (reward + Gamma * self.values[new_state]) # Bellman equation
        return value

    def select_action(self, board: np.ndarray) -> Action:
        state = self._state_tuple(board)
        best_action, best_value = 0, float('-inf')
        for action in range(9):
            value = self.calc_action_value(state, action)
            if value > best_value:
                best_value, best_action = value, action
        return best_action

    def play_episode(self) -> float:
        total_reward = 0.0
        board, _, _ = self.env.reset()
        while True:
            action = self.select_action(board)
            state = self._state_tuple(board)
            board, reward, done = self.env.step(action)
            new_state = self._state_tuple(board)
            self.rewards[(state, action, new_state)] = reward
            self.transitions[(state, action)][new_state] += 1
            total_reward += reward

            if done:
                break
        return total_reward

    def value_iteration(self):
        for board in self.env.generate_all_states():
            state = self._state_tuple(board)
            action_vals = [self.calc_action_value(state, action) for action in range(9)]
            self.values[state] = max(action_vals) if action_vals else 0.0

if __name__ == "__main__":
    agent = TicTacToeAgent()
    writer = SummaryWriter(comment="tic-tac-toe-q")
    best_reward = float('-inf')
    iter_no = 0

    while True:
        iter_no += 1
        agent.play_n_random_steps(100)
        agent.value_iteration()

        # test performance
        avg_reward = 0.0
        for _ in range(TEST_EPISODE):
            avg_reward += agent.play_episode()
        avg_reward /= TEST_EPISODE

        writer.add_scalar("avg_reward", avg_reward, iter_no)
        if avg_reward > best_reward:
            print(f"Iter {iter_no}: new best avg reward {avg_reward:.3f}")
            best_reward = avg_reward
        if best_reward >= 1.0:
            print("Solved!")
            print(agent.env)
            break

    writer.close()
