# baselineTeam.py
# ---------------
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


# baselineTeam.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from captureAgents import CaptureAgent
import distanceCalculator
import random, time, util, sys
from game import Directions, Actions
from capture import GameState
import game
from util import nearestPoint


#################
# Team creation #
#################

# def createTeam(firstIndex, secondIndex, thirdIndex, isRed,
def createTeam(indices, isRed, first='OffensiveReflexAgent', second='DefensiveReflexAgent'):
    """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
    agents = [eval('VermillionAgent' if (index // 2) % 2 == 0 else 'DefensiveReflexAgent')(index) for index in indices]
    return agents
    # return [eval(first)(firstIndex), eval(first)(secondIndex), eval(second)(thirdIndex)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
  A base class for reflex agents that chooses score-maximizing actions
  """

    def registerInitialState(self, gameState):
        self.start = gameState.getAgentPosition(self.index)
        CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
        """
    Picks among the actions with the highest Q(s,a).
    """
        actions = gameState.getLegalActions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(gameState, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        maxValue = max(values)
        bestActions = [a for (a, v) in zip(actions, values) if v == maxValue]

        foodLeft = len(self.getFood(gameState).asList())

        if foodLeft <= 20:
            bestDist = 9999
            for action in actions:
                successor = self.getSuccessor(gameState, action)
                pos2 = successor.getAgentPosition(self.index)
                dist = self.getMazeDistance(self.start, pos2)
                if dist < bestDist:
                    bestAction = action
                    bestDist = dist
            return bestAction

        return random.choice(bestActions)

    def getSuccessor(self, gameState, action):
        """
    Finds the next successor which is a grid position (location tuple).
    """
        successor = gameState.generateSuccessor(self.index, action)
        pos = successor.getAgentState(self.index).getPosition()
        if pos != nearestPoint(pos):
            # Only half a grid position was covered
            return successor.generateSuccessor(self.index, action)
        else:
            return successor

    def evaluate(self, gameState, action):
        """
    Computes a linear combination of features and feature weights
    """
        features = self.getFeatures(gameState, action)
        weights = self.getWeights(gameState, action)
        return features * weights

    def getFeatures(self, gameState, action):
        """
    Returns a counter of features for the state
    """
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)
        features['successorScore'] = self.getScore(successor)
        return features

    def getWeights(self, gameState, action):
        """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
        return {'successorScore': 1.0}


class DefensiveReflexAgent(ReflexCaptureAgent):
    """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # Computes whether we're on defense (1) or offense (0)
        features['onDefense'] = 1
        if myState.isPacman: features['onDefense'] = 0

        # Computes distance to invaders we can see
        enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
        invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
        features['numInvaders'] = len(invaders)
        if len(invaders) > 0:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
            features['invaderDistance'] = min(dists)

        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1

        return features

    def getWeights(self, gameState, action):
        return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


'''
Vermillion is an offensive agent that evades enemy chasers
Vermillion gets scared very easily when he is holding food,
            depending on the circumstances he will flyToSafeZone
Vermillion will only attack invaders if they are directly in front of him
'''


class VermillionAgent(ReflexCaptureAgent):
    def getFeatures(self, gameState, action):
        features = util.Counter()
        successor = self.getSuccessor(gameState, action)

        # getting the state and the position
        myState = successor.getAgentState(self.index)
        myPos = myState.getPosition()

        # getting the coordinates
        xPos, yPos = myState.getPosition()
        # getting relative distance
        relX, relY = Actions.directionToVector(action)
        # next move ahead
        nextX, nextY = int(xPos + relX), int(yPos + relY)

        # creating sections here
        if self.index > 2:
            section = 1
        else:
            section = self.index + 1
        # get opponent food in the list
        food = self.getFood(gameState)
        appendFood = []
        foodList = food.asList()
        # get the walls around
        walls = gameState.getWalls()

        foodEaten = gameState.getAgentState(self.index).numCarrying

        # safezone cooridinat
        upperSafeZoneDist = float(min([self.getMazeDistance(myPos, (16, 16))]))
        lowerSafeZoneDist = float(min([self.getMazeDistance(myPos, (21, 6))]))

        # is your agent PacMan?
        isPacman = myState.isPacman
        # gets all your opponents
        opponents = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
        # gets the opponents pacman in your zone
        invaders = [opponent for opponent in opponents if opponent.isPacman and opponent.getPosition() != None]
        # gets the ghosts your opponents zone
        chasers = [opponent for opponent in opponents if not (opponent.isPacman) and opponent.getPosition() != None]

        # if action is stop then set the feature stop
        if action == Directions.STOP: features['stop'] = 1

        # checking the food and splitting the maze into two for the team
        if self.index < 3:
            if len(foodList) > 0:
                for foodX, foodY in foodList:
                    # uses the grid's upper bound of the y-axis to find food, divides the maze for each vermillion
                    if (foodY > section * walls.height / 2 and foodY < (section + 1) * walls.height / 2):
                        appendFood.append((foodX, foodY))
                if len(appendFood) == 0:
                    appendFood = foodList
                if min([self.getMazeDistance(myPos, food) for food in appendFood]) is not None:
                    features['distanceToFood'] = float(min([self.getMazeDistance(myPos, food) for food in appendFood])) / (
                            walls.width * walls.height)
        elif self.index >= 3:
            if len(foodList) > 0:
                for foodX, foodY in foodList:
                    # uses the grid's lower bound of the y-axis to find food, divides the maze for each vermillion
                    if (foodY < section * walls.height / 2):
                        appendFood.append((foodX, foodY))
                if len(appendFood) == 0:
                    appendFood = foodList
                if min([self.getMazeDistance(myPos, food) for food in appendFood]) is not None:
                    features['distanceToFood'] = float(min([self.getMazeDistance(myPos, food) for food in appendFood])) / (
                            walls.width * walls.height)




        # features for attacking invaders
        for invader in opponents:
            if invader.getPosition() != None:
                if (nextX, nextY) == invader.getPosition:
                    features['attackInvader'] = 10
                elif (nextX, nextY) in Actions.getLegalNeighbors(invader.getPosition(), walls):
                    features['invaderAhead'] += float(min([self.getMazeDistance(myPos, invader.getPosition()) for invader in opponents]))
                    features['attackInvader'] += 2

        # checking if there is are chasers around

        for ghosts in chasers:
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in chasers]

            if isPacman:

                # multiple conditions to see if enemy is close enough to flee
                if (nextX, nextY) in Actions.getLegalNeighbors(ghosts.getPosition(), walls) or \
                        (nextX, nextY) == ghosts.getPosition() or \
                        min(dists) <= 3:
                    dists = [self.getMazeDistance(myPos, a.getPosition()) for a in chasers]
                    features['flee'] = max(dists) * 1500
                    print
                    "Fleeing"

                if self.index < 3:
                    if foodEaten > 5 and min(dists) <= 5 or\
                            foodEaten > 10 and min(dists) <= 11 or\
                            foodEaten > 17 and self.getScore(gameState) < 40 or \
                            foodEaten > 5 and self.getScore(gameState) >= 40 or \
                            foodEaten > 5 and upperSafeZoneDist <= 8:
                        features['flyToSafeZone'] = upperSafeZoneDist * 500
                elif self.index >= 3:
                    if foodEaten > 5 and min(dists) <= 5 or\
                            foodEaten > 10 and min(dists) <= 11 or\
                            foodEaten > 17 and self.getScore(gameState) < 40 or \
                            foodEaten > 5 and self.getScore(gameState) >= 40 or \
                            foodEaten > 5 and lowerSafeZoneDist <= 8:
                        features['flyToSafeZone'] = lowerSafeZoneDist * 500

            else:

                # during ghost mode always attack pacman by taking minimum distance

                print
                "ghosts position", ghosts.getPosition()
                features['attackPacMan'] = min([self.getMazeDistance(myPos, a.getPosition()) for a in chasers])


        return features

    def getWeights(self, gameState, action):
        return {'distanceToFood': -1, 'attackPacMan': -20, 'stop': -50, 'flee': -40,
                'attackInvader': -1, 'invaderAhead': -1, 'eatFood': 1, 'flyToSafeZone': -1}


