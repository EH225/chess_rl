# -*- coding: utf-8 -*-
"""
This module contains helper functions for running agent evaluations.
"""
import sys, os

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PARENT_DIR)

import asyncio, subprocess

asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

import chess
import chess.engine
from typing import Callable

STOCKFISH_PATH = rf"{os.path.join(PARENT_DIR, 'stockfish/stockfish-windows-x86-64-avx2.exe')}"


def compute_move_loss(engine: chess.engine.SimpleEngine, board: chess.Board, move: chess.Move,
                      time_limit: float = 0.05, mate_score: int = 30000):
    """
    Given a starting chess board game state, this function computes the centipawn loss estimated by an oracle
    engine of making a particular move from the current position for the player whose move is next. We have
    an engine estimate the advantage for the current player and compute how much it changes with the next move
    it makes. Generally, a good player should have a similar (or better) advantage after making its next move.
    A large drop in advantage (an estimate of win probability measured in centipawn units) indicates a blunder
    on behalf of the player making the move.

    If the engine cannot return a numerical score, then a 0 is returned.
    """
    # Have the oracle engine evaluate the board before the move is made
    info_before = engine.analyse(board, chess.engine.Limit(time=time_limit))
    score_before = info_before["score"]
    if board.turn == chess.WHITE:
        eval_before = score_before.white().score(mate_score=mate_score)
    else:
        eval_before = score_before.black().score(mate_score=mate_score)

    # Apply the move to the board
    board.push(move)

    # Have the oracle engine evaluate the board again after the move is made
    info_after = engine.analyse(board, chess.engine.Limit(time=time_limit))
    score_after = info_after["score"]
    if board.turn == chess.WHITE:
        eval_after = score_after.white().score(mate_score=mate_score)
    else:
        eval_after = score_after.black().score(mate_score=mate_score)

    # Backtrack to restore the board so that no move is actually made
    board.pop()

    # If either evaluation couldn't be converted to an integer, return a 0 instead
    if eval_before is None or eval_after is None:
        return 0

    # Compute the centipawn loss of making this move i.e. how much it reduced the player's win advantage
    # Note that this is only defined for moves that make the board position worse for the player and is
    # something that should be minimized
    return max(eval_before - eval_after, 0)


def evaluate_agent_game(agent_move_func: Callable, max_moves: int = 300, return_move_stack: bool = False):
    """
    Run self-play chess game evaluation using Stockfish as an evaluation oracle to compute Average Centipawn
    Loss (ACPL). Games are capped at a total of max_moves.

    :param agent_move_func: A callable function that when passed a chess.Board object, returns the agent's
        recommended next move as a chess.Move object.
    :param max_moves: An integer denoting the max number of moves allowed per game.
    :param return_move_stack: If True, then the move stack of the game is returned as well.
    :return: A list of centipawn losses for all moves during a self-play chess game.
    """
    # Load the stockfish chess engine from local cache
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH, stderr=subprocess.DEVNULL)
    board = chess.Board()  # Begin a new chess game from the start
    move_losses = []  # Collect the centipawn losses for each move made by the RL agent

    move_counter = 0
    while move_counter < max_moves and not board.is_game_over():
        move = agent_move_func(board)  # Get the next move to play from the agent using the board as input
        loss = compute_move_loss(engine, board, move)  # Evaluate the loss of that move, compare the current
        # board state vs the next board state win probability for the player to move according to the oracle
        move_losses.append(loss)
        board.push(move)  # Make the actual move and continue to loop
        move_counter += 1

    engine.quit()  # Close down the oracle evaluator chess engine (stockfish)
    return (move_losses, board.move_stack) if return_move_stack is True else move_losses

# def random_agent(board):
#     return random.choice(list(board.legal_moves))

## TODO: Add an elo run tournament set of functions that can give us Elo scores more accurately
## and also potentially have my agents play against on another
