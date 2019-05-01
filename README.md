#Nine-Board Tic-Tac-Toe AI Player.

[![Build Status](https://travis-ci.com/martinlecs/comp3411-tic-tac-toe.svg?token=kGtS9cVnuVjkP2dryxvf&branch=master)](https://travis-ci.com/martinlecs/comp3411-tic-tac-toe)

## Introduction
This game is played on a 3 x 3 array of 3 x 3 Tic-Tac-Toe boards. The first move is made by placing an X in a randomly chosen cell of a randomly chosen board. After that, the two players take turns placing an O or X alternately into an empty cell of the board corresponding to the cell of the previous move. (For example, if the previous move was into the upper right corner of a board, the next move must be made into the upper right board.)

The game is won by getting three-in-a row either horizontally, vertically or diagonally in one of the nine boards. If a player is unable to make their move (because the relevant board is already full) the game ends in a draw.


## Implementation Details

This AI uses the standard minimax search to find the best possible move for the player with alpha beta pruning optimisation to reduce the number of nodes expanded.
Our heuristic aims to maximise the number of adjacent values for the current player (See Heuristic.py for more info).
(ie. If I'm player X I want to maximise the number of adjacent Xs on every board).

We use a GameTeeNode Class to represent each of our game states. It contains a reference to its global state,
the current board in play, its parent board (if it has a parent), as well as alpha value generated for that state.

We have a Game Class that is used to perform all game-related actions such as:
  1) Checking if a state is a terminal state
  2) Converting between hash values and boards
  3) Generate moves

A number of optimisations were used to enable the AI to go as deep as possible within the time:
  1) Hashing every possible board state
      - Can go from hash value -> board state (represented by a 1x10 numpy array)
      - Can go from board state (numpy array) -> hash value
  2) Precalculate the heuristic for every board state and use the hash value (generated above) whenever we need to
      refer to them.
  3) Generate all possible win states for a Tic-Tac-Toe board and store these in a hash. We use the board's hash value
      to check if the board is a win state.
  *) All the above are stored in pickle files and loaded in during start-up.

These optimisation enabled our AI to reach depth 7 with < 1 second per call to our alpha beta search function.

Micro optimisations within Python:
  1) When we generate children, we don't generate all the children at once. We use a generator to create children on
      the fly and evaluate them sequentially. We minimise copying of states by making a move and then undoing it right
      after.
  2)  We avoid using loops, preferring list comprehensions or better yet, built-ins. Anything that does involve loops
      has been cached.

## How to Run

1. Install requirements with `pip3 install -r requirements.txt`.
2. `cd` into `/src` and run `make all`.
3. Run the `play.sh` script in the root directory to run a game.

__Modifying Heuristic__
1. Edit `player/Heuristic.py`.
2. Modify class as needed.

__Adding Tests__
1. Look up pytest.

