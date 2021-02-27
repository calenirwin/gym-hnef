# Notes



# Neural Network:
Trained to predict the policy (p -> pi) and value (v -> the winner of the game)

Evaluated by playing against the current best NN, both players use MCTS to choose their moves and
    while the NN's evaluate the moves. 

The new net must win at least 55% to be considered the best.

Input: 11x11x3?
11x11 binary array which contains the locations of black pieces
11x11 binary array which contains the locations of white pieces
Either a binary number or array that indicates which turn it is.

Output: p and v
p: array which indicates the precentages of picking each possible move
v: estimation of current player winning precentage


# MCTS:

Every action has four values:
N: How many times has this action been taken?
W: Total value for the next state
Q: Mean value for the next state
P: Probability of picking this action

Run (1600? times):

1. Pick action that maximizes Q + U(N,P)
2. Keep going until a leaf node is reached, that state is evaluated by the NN and the p becomes the 
    new P value
3. Update tree:
N = N+1
W = W+v
Q = W/N

We can pick actions:

Deterministically (evaluation): Action with the highest N
Stochastically (exploration): Randomly where pi ~ N^(1/Tau) where Tau controls exploration
*
*
*