# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        new_score = successorGameState.getScore()
        min_food_dist = min(
            [manhattanDistance(newPos, foodPos) for foodPos in newFood.asList()] or [0]
        )
        food_score = 1.0 / (min_food_dist + 1) * 10

        min_ghost = min(
            [
                (
                    manhattanDistance(newPos, ghostState.getPosition()),
                    ghostState.scaredTimer,
                )
                for ghostState in newGhostStates
            ]
        )
        min_ghost_dist, min_ghost_timer = min_ghost

        ghost_score = 0
        if min_ghost_dist < 5 and min_ghost_timer == 0:
            ghost_score -= 500
        elif min_ghost_timer > 0:
            ghost_score = 500 / (min_ghost_dist + 1)

        return new_score + food_score + ghost_score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(game_state, depth, agent_index):
            if depth == 0 or game_state.isWin() or game_state.isLose():
                return self.evaluationFunction(game_state)

            legal_actions = game_state.getLegalActions(agent_index)
            if agent_index == 0:  # index 0 -> PacMan player
                max_score = float("-inf")
                for action in legal_actions:
                    new_game_state = game_state.generateSuccessor(0, action)
                    curr_score = minimax(new_game_state, depth, 1)
                    max_score = max(max_score, curr_score)

                return max_score

            min_score, num_agents = float("inf"), game_state.getNumAgents()
            for action in legal_actions:
                new_game_state = game_state.generateSuccessor(agent_index, action)
                curr_score = (
                    minimax(new_game_state, depth, agent_index + 1)
                    if agent_index != num_agents - 1
                    else minimax(new_game_state, depth - 1, 0)
                )
                min_score = min(min_score, curr_score)

            return min_score

        # since minimax returns a number we need the below to find the associated action
        max_score, max_action = float("-inf"), None
        for action in gameState.getLegalActions(0):
            new_game_state = gameState.generateSuccessor(0, action)
            curr_score = minimax(new_game_state, self.depth, 1)
            if curr_score > max_score:
                max_score = curr_score
                max_action = action

        return max_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        def minimax(game_state, depth, agent_index, alpha, beta):
            if depth == 0 or game_state.isWin() or game_state.isLose():
                return self.evaluationFunction(game_state)

            legal_actions = game_state.getLegalActions(agent_index)
            if agent_index == 0:  # index 0 -> PacMan player
                max_score = float("-inf")
                for action in legal_actions:
                    new_game_state = game_state.generateSuccessor(0, action)
                    curr_score = minimax(new_game_state, depth, 1, alpha, beta)
                    max_score = max(max_score, curr_score)

                    if max_score > beta:
                        return max_score

                    alpha = max(alpha, max_score)

                return max_score

            min_score, num_agents = float("inf"), game_state.getNumAgents()
            for action in legal_actions:
                new_game_state = game_state.generateSuccessor(agent_index, action)
                curr_score = (
                    minimax(new_game_state, depth, agent_index + 1, alpha, beta)
                    if agent_index != num_agents - 1
                    else minimax(new_game_state, depth - 1, 0, alpha, beta)
                )
                min_score = min(min_score, curr_score)

                if min_score < alpha:
                    return min_score

                beta = min(beta, min_score)

            return min_score

        # since minimax returns a number we need the below to find the associated action
        max_score, max_action = float("-inf"), None
        alpha, beta = float("-inf"), float("inf")

        for action in gameState.getLegalActions(0):
            new_game_state = gameState.generateSuccessor(0, action)
            curr_score = minimax(new_game_state, self.depth, 1, alpha, beta)

            if curr_score > max_score:
                max_score = curr_score
                max_action = action

            alpha = max(alpha, max_score)

        return max_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def expectimax(game_state, depth, agent_index):
            if depth == 0 or game_state.isWin() or game_state.isLose():
                return self.evaluationFunction(game_state)

            legal_actions = game_state.getLegalActions(agent_index)
            if agent_index == 0:  # index 0 -> PacMan player
                max_score = float("-inf")
                for action in legal_actions:
                    new_game_state = game_state.generateSuccessor(0, action)
                    curr_score = expectimax(new_game_state, depth, 1)
                    max_score = max(max_score, curr_score)

                return max_score

            exp_score, num_agents = 0, game_state.getNumAgents()
            for action in legal_actions:
                new_game_state = game_state.generateSuccessor(agent_index, action)
                curr_score = (
                    expectimax(new_game_state, depth, agent_index + 1)
                    if agent_index != num_agents - 1
                    else expectimax(new_game_state, depth - 1, 0)
                )
                exp_score += curr_score * 1 / len(legal_actions)

            return exp_score

        # since expectimax returns a number we need the below to find the associated action
        max_score, max_action = float("-inf"), None
        for action in gameState.getLegalActions(0):
            new_game_state = gameState.generateSuccessor(0, action)
            curr_score = expectimax(new_game_state, self.depth, 1)
            if curr_score > max_score:
                max_score = curr_score
                max_action = action

        return max_action


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return 10000
    if currentGameState.isLose():
        return -10000

    ghost_states = currentGameState.getGhostStates()
    score = currentGameState.getScore()
    food = currentGameState.getFood()
    pacman_pos = currentGameState.getPacmanPosition()

    min_food_dist = min(
        [manhattanDistance(pacman_pos, foodPos) for foodPos in food.asList()] or [0]
    )
    food_score = 1.0 / (min_food_dist + 1) * 10

    min_ghost = min(
        [
            (
                manhattanDistance(pacman_pos, ghost_state.getPosition()),
                ghost_state.scaredTimer,
            )
            for ghost_state in ghost_states
        ]
    )
    min_ghost_dist, min_ghost_timer = min_ghost

    ghost_score = 0
    if min_ghost_dist < 5 and min_ghost_timer == 0:
        ghost_score -= 1000
    elif min_ghost_timer > 0:
        ghost_score = 1000 / (min_ghost_dist + 1)

    return score + food_score + ghost_score


# Abbreviation
better = betterEvaluationFunction
