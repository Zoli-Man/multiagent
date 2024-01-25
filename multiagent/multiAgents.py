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
#yosiiiii
class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
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
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
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
        ## dist to closest food
        food_dist = 99999999999
        for foodPos in newFood.asList():
            food_dist = min(food_dist, manhattanDistance(newPos, foodPos))
        food_eat = (len(currentGameState.getFood().asList()) - len(newFood.asList())) * 10000
        ## dist to closest ghost
        closet_ghost = 99999999999
        for ghostState in newGhostStates:
            closet_ghost = min(closet_ghost, manhattanDistance(newPos, ghostState.getPosition()))
        if closet_ghost < 2:
            return -99999999999



        return 1000/food_dist + food_eat

def scoreEvaluationFunction(currentGameState):
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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """


    def getAction(self, gameState):
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
        def gomax(gameState, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            maxval = -99999999999
            legalMoves = gameState.getLegalActions(0)
            for move in legalMoves:
                successor = gameState.generateSuccessor(0, move)
                maxval = max(maxval, gomin(successor, depth, 1))
            return maxval
        def gomin(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            minval = 99999999999
            legalMoves = gameState.getLegalActions(agentIndex)
            for move in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, move)
                if agentIndex == gameState.getNumAgents() - 1:
                    # next agent is pacman
                    minval = min(minval, gomax(successor, depth + 1))
                else:
                    # next agent is a ghost
                    minval = min(minval, gomin(successor, depth, agentIndex + 1))
            return minval




        legalMoves = gameState.getLegalActions(0)
        best_move = None, -99999999999 # (move, score)
        for move in legalMoves:
            secseccesor = gameState.generateSuccessor(0, move)
            score = gomin(secseccesor, 0, 1)
            if score > best_move[1]:
                best_move = move, score
        return best_move[0]
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        ALPHA = -99999999999
        BETA = 99999999999
        def gomax(gameState, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            maxval = -99999999999
            legalMoves = gameState.getLegalActions(0)
            for move in legalMoves:
                successor = gameState.generateSuccessor(0, move)
                maxval = max(maxval, gomin(successor, depth, 1, alpha, beta))
                if maxval > beta:
                    return maxval
                alpha = max(alpha, maxval)
            return maxval
        def gomin(gameState, depth, agentIndex, alpha, beta):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            minval = 99999999999
            legalMoves = gameState.getLegalActions(agentIndex)
            for move in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, move)
                if agentIndex == gameState.getNumAgents() - 1:
                    # next agent is pacman
                    minval = min(minval, gomax(successor, depth + 1, alpha, beta))
                    if minval < alpha:
                        return minval
                    beta = min(beta, minval)
                else:
                    # next agent is a ghost
                    minval = min(minval, gomin(successor, depth, agentIndex + 1, alpha, beta))
                    if minval < alpha:
                        return minval
                    beta = min(beta, minval)
            return minval

        legalMoves = gameState.getLegalActions(0)
        best_move = None, -99999999999
        for move in legalMoves:
            secseccesor = gameState.generateSuccessor(0, move)
            score = gomin(secseccesor, 0, 1, ALPHA, BETA)
            if score > best_move[1]:
                best_move = move, score
            ALPHA = max(ALPHA, score)
        return best_move[0]
class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def gomax(gameState, depth):
            curr_depth = depth + 1
            if gameState.isWin() or gameState.isLose() or curr_depth == self.depth:
                return self.evaluationFunction(gameState)
            maxval = -99999999999
            legalMoves = gameState.getLegalActions(0)
            for move in legalMoves:
                successor = gameState.generateSuccessor(0, move)
                maxval = max(maxval, gorandom(successor, curr_depth, 1))
            return maxval

        def gorandom(gameState, depth, agentIndex):
            if gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            legalMoves = gameState.getLegalActions(agentIndex)
            num_of_moves = len(legalMoves)
            sum_of_values = 0
            for move in legalMoves:
                successor = gameState.generateSuccessor(agentIndex, move)
                if agentIndex == gameState.getNumAgents() - 1:
                    # next agent is pacman
                    expected_value = gomax(successor, depth)
                else:
                    # next agent is a ghost
                    expected_value = gorandom(successor, depth, agentIndex + 1)
                sum_of_values += expected_value
            return sum_of_values / num_of_moves



        legalMoves = gameState.getLegalActions(0)
        best_move = None, -99999999999  # (move, score)
        for move in legalMoves:
            secseccesor = gameState.generateSuccessor(0, move)
            score = gorandom(secseccesor, 0, 1)
            if score > best_move[1]:
                best_move = move, score
        return best_move[0]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # dist to ghost from the current state
    ghostPos = []
    for ghost in newGhostStates:
        ghostPos.append(ghost.getPosition())
    ghost_dist = []
    for pos in ghostPos:
        ghost_dist.append(manhattanDistance(newPos, pos))

    #  dist to food from the current state
    food_list = newFood.asList()
    food_dist = []
    for pos in food_list:
        food_dist.append(manhattanDistance(newPos, pos))



    score = currentGameState.getScore()
    sum_scard = sum(newScaredTimes)
    sum_ghoste_dist = sum(ghost_dist)
    # make sure that the pacman will get closer to the food
    if sum(food_dist) > 0:
        score += 1.0 / sum(food_dist)


    # make sure that the pacman will eat the food
    score += len(newFood.asList(False))

    # if the ghost is scared, the pacman will try to eat it
    if sum_scard > 0:
        score += sum_scard  + (-1 * sum_ghoste_dist)
    # if the ghost is not scared, the pacman will try to avoid it
    else:
        for ghost_dist in ghost_dist:
            if ghost_dist < 1:
                return float("-inf")


    return score






# Abbreviation
better = betterEvaluationFunction
