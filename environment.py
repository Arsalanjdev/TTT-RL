"""
A simple tic-tac-toe game environment designed for RL experiments.
Inspired by gym environments, it returns a tuple of (state, reward, is_done) each step by an action.
"""
from typing import Tuple

import numpy as np
from dataclasses import dataclass


@dataclass
class EnvResult:
    """
    The result of a step in the environment: the new state, the reward and is_done boolean flag.
    """
    state: np.ndarray
    reward: int
    is_done: bool

class TicTacToeEnv:

    def __init__(self):
        self.board = np.zeros((3, 3), dtype=int)
        self.current_player = np.random.randint(1, 3)  # 1 for the environment, 2 for the RL agent
        self.is_done = False
        self.reward = 0

    def reset(self) -> EnvResult:
        """
        Resets and initiates the board
        :return:
        """
        self.board = np.zeros((3,3),dtype=int)
        self.is_done = False
        self.current_player = np.random.randint(1,3)  # 1 for the environment, 2 for the RL agent
        if self.current_player == 1: # If it's the environment's turn, let it play a step
            return self.step()
        self.reward = 0
        return EnvResult(self.board,self.reward,self.is_done)

    def step(self,action: int) -> EnvResult:
        """
        The step action of the environment that executes the action of the RL agent and returns the next board state and
        its reward.
        :param action: an Integer that represents the position of board that is getting marked
        :return:
        """
        self._mark(action)
        self._env_action()
        return EnvResult(self.board, self.reward, self.is_done)



    def _check_done(self) -> int:
        """
        checks if the game is over.
        :return: -2 if the environment won, 1 if the agent won, 0 if it's ongoing and -1 if the game is a tie.
        """

        #checking rows and columns
        for i in range(3):
            if np.all(self.board[i,:] == 1):
                return 1
            if np.all(self.board[i,:] == 2):
                return -2

            if np.all(self.board[:,i] == 1):
                return 1
            if np.all(self.board[:,i] == 2):
                return -2

        #checking diagonals
        if np.all(np.diag(self.board) == 1):
            return 1
        if np.all(np.diag(self.board) == 2):
            return -2

        if np.all(np.diag(np.fliplr(self.board)) == 1):
            return 1
        if np.all(np.diag(np.fliplr(self.board)) == 2):
            return -2

        # checking if all cells are filled
        if np.any(self.board == 0):
            return 0
        return -1



    def _env_action(self):
        """
        Execution of an action by the environment which is randomly decided.
        :return:
        """
        if self.is_done:
            return
        while True:
            choice = np.random.randint(0,10)
            if self._is_legal(choice):
                self._mark(choice)
                break


    def _is_legal(self,action: int) -> bool:
        """
        Check if the action provided (the position in the board) is legal.
        :param action: an integer that ranges from 0 to 9 for each cell on the board.
        :return: True if the action is legal, False otherwise.
        """
        row, col = divmod(action,3)
        return self.board[row,col] == 0

    def _mark(self,action: int):
        """
        Marks the given position as marked by the current player.
        :param action: an Integer that represents the position of board that is getting marked.
        """
        if not self._is_legal(action):
            self.reward = -10
            self.is_done = True
        else:
            row, col = divmod(action, 3)
            self.board[row, col] = self.current_player
            self.current_player = 1 if self.current_player == 2 else 2
            self.reward = self._check_done()
            self.is_done = False
