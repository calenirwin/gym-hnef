# AlphaZero notes

## Neural Network:
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


## MCTS:

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


## Evaluation
Evaluate against random agent
Observe distributions of Win/Draw/Loss
Observe distributions of Draws

## Baseline agents
Random
Simple rule based 

## Training

To ensure that our agents don't converge on a strategy that only works against a weak opponent we will train one agent at a time and as soon as it improves we train the other one.

# Game balance of the variants of Hnefatafl considered

## Table created from data taken from the Hnefatafl Steam app


| Variant        | Black wins           | Draws  | White wins | Balance | 
| ------------- |-------------| -----| --- | ---  |
| Historical Hnefatafl      | 260 | 1 | 274 | 1.05 | 
| Copenhagen Hnefatafl     | 895      |  25 | 1704 | 1.89 |


## Table taken from the largest database of Hnefatafl matches

| Variant        | Overall           | Full matches  | Strong players | Strong players full matches | Average |
| ------------- |-------------| -----| --- | ---  | ---  |
| Historical Hnefatafl      | -1.15 | -1.14 | -1.23 | -1.21 | -1.18 |
| Copenhagen Hnefatafl     | 1.41      |   1.47 | 1.39 | 1.46 | 1.46 |

Positive means white wins more often, negative means black wins more often.

Full matches mean pairs of matches where players take two games, one as each side.

Draws were counted as half wins.

Link: 
http://aagenielsen.dk/tafl_balances.php


# Thesis notes

## Evaluation metrics

### Evaluation is split into two measures:

* Intelligence: Performance against a random player
* Generality: Performance against themselves

Maybe we can try having the attacker agent play as the defender and vice versa to see how their strategies change?


### State space complexity

|     |  Historical Hnefatafl   |  Othello |  Chess   |  Go |
| --- | --- | --- | --- | --- |
|  Upper bound on state-space complexity |   1.4 * 10^27  |  10^28   |  10^43 or 10^50   |   2 * 10^170  |


http://ai.unibo.it/sites/ai.unibo.it/files/Complexity_of_Tablut_0.pdf

# General thoughts

One thing is that it takes time to calculate all of the valid moves and we need such computations each time we change states. For Go, finding the legal moves is the same as checking what board tiles are empty, this is more complicated for Hnefatafl.

Instead of having it be a part of the state, I think we should have the list of valid moves as a variable in the hnef_game class or hnef_env class and then update it each time we change states.
