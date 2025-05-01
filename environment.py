"""
A simple tic-tac-toe game environment designed for RL experiments.
Inspired by gym environments, it returns a tuple of (state, reward, is_done) each step by an action.
"""
import random
from itertools import product
from typing import Tuple, List

import numpy as np

State = np.ndarray
Reward = int
IsDone = bool
EnvResult = Tuple[State, Reward, IsDone]

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
        self.board[:] = 0
        self.is_done = False
        self.current_player = np.random.randint(1,3)  # 1 for the environment, 2 for the RL agent
        self.reward = 0
        if self.current_player == 1: # If it's the environment's turn, let it play a step
            self._env_action()
        return self.board,self.reward,self.is_done

    def step(self,action: int) -> EnvResult:
        """
        The step action of the environment that executes the action of the RL agent and returns the next board state and
        its reward.
        :param action: an Integer that represents the position of board that is getting marked
        :return:
        """
        self._mark(action)
        if not self.is_done:
            self._env_action()
        return self.board,self.reward,self.is_done

    def generate_all_states(self):
        states: List[np.ndarray] = []
        for cells in product((0,1,2),repeat=9):
            board = np.array(cells).reshape(3,3)
            count_x = np.sum(board == 1)
            count_o = np.sum(board == 2)
            if abs(count_x - count_o) <= 1:
                states.append(board)
        return states


    def _check_done(self) -> int:
        """
        checks if the game is over.
        :return: --1 if the environment won, 1 if the agent won, 0 if it's ongoing or a tie.
        """

        #checking rows and columns
        for i in range(3):
            if np.all(self.board[i,:] == 1):
                return 1
            if np.all(self.board[i,:] == 2):
                return -1

            if np.all(self.board[:,i] == 1):
                return 1
            if np.all(self.board[:,i] == 2):
                return -1

        #checking diagonals
        if np.all(np.diag(self.board) == 1):
            return 1
        if np.all(np.diag(self.board) == 2):
            return -1

        if np.all(np.diag(np.fliplr(self.board)) == 1):
            return 1
        if np.all(np.diag(np.fliplr(self.board)) == 2):
            return -1

        # checking if all cells are filled
        # if np.any(self.board == 0):
        #     return 0
        # return -1
        return 0



    def _env_action(self):
        """
        Execution of an action by the environment which is randomly decided.
        :return:
        """
        if self.is_done:
            return
        choice = np.random.randint(0,9)
        while not self._is_legal(choice):
            choice = np.random.randint(0,9)
        self._mark(choice)

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
            self.reward = -1
            self.is_done = True
        else:
            row, col = divmod(action, 3)
            self.board[row, col] = self.current_player
            self.current_player = 1 if self.current_player == 2 else 2
            self.reward = self._check_done()
            self.is_done = (self.reward != 0) or not np.any(self.board == 0)

    def __str__(self):
        """
        :return: The string representation of the board. X indicates the player marks and O indicates the opponent marks.
        """
        board_str = ""
        for row in self.board:
            board_str += " | ".join(['X' if cell == 2 else 'O' if cell == 1 else '.' for cell in row]) + "\n"
            board_str += "-" * 9 + "\n"  # Line separator between rows

        return board_str.strip()  # Remove the last separator line.