import numpy as np
import random

class Easy21():

    def __init__(self):
        self.minCardValue, self.maxCardValue = 2, 11
        self.dealerUpperBound = 16
        self.gameLowerBound, self.gameUpperBound = 0, 21
        self.init_deck()

    @classmethod
    def actionSpace(self):
        return (0, 1)

    def init_deck(self):
        self.deck = 4 * list(range(2,12))
        [self.deck.append(10) for _ in range (12)]
        # self.deck = random.shuffle(self.deck)

    def initGame(self):
        player = np.random.randint(self.minCardValue, self.maxCardValue + 1) + np.random.randint(self.minCardValue, self.maxCardValue + 1) # two cards for gambler
        while player == 22: # init again for case A+A (in our case is 22 - invalid)
            player = np.random.randint(self.minCardValue, self.maxCardValue + 1) + np.random.randint(self.minCardValue,
                                                                                                     self.maxCardValue + 1)  # two cards for gambler
        return (player,
                np.random.randint(self.minCardValue, self.maxCardValue + 1))

    def draw(self):
        value = np.random.randint(self.minCardValue, self.maxCardValue + 1)

        value = np.random.choice(np.array(self.deck))
        self.deck.remove(value)

        # if np.random.random() <= 1 / 3:
        #     return -value
        # else:
        return value

    def step(self, playerValue, dealerValue, action):

        assert action in [0, 1], "Expection action in [0, 1] but got %i" % action

        if action == 0:

            playerValue += self.draw()

            # check if player busted
            if not (self.gameLowerBound < playerValue <= self.gameUpperBound):
                reward = -1
                terminated = True

            else:
                reward = 0
                terminated = False

        elif action == 1:
            terminated = True

            while self.gameLowerBound < dealerValue < self.dealerUpperBound:
                dealerValue += self.draw()

            # check if dealer busted // playerValue greater than dealerValue
            if not (self.gameLowerBound < dealerValue <= self.gameUpperBound) or playerValue > dealerValue:
                reward = 1

            elif playerValue == dealerValue:
                reward = 0

            elif playerValue < dealerValue:
                reward = -1

        return playerValue, dealerValue, reward, terminated