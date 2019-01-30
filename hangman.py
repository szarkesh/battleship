import numpy as np
from collections import defaultdict
import numpy.random as random
import math
import pdb
import string
import operator
from copy import deepcopy
phrase = "banana"
def load_corpus():
    with open("words.txt") as f:
        content = f.readlines()
    # you may also want to remove whitespace characters like `\n` at the end of each line
    content = [x.strip() for x in content]
    return content
words = load_corpus()


def playGame(currSitch):
    while (isGameOver(currSitch)==False):
        guess = mlGuesser(currSitch)
        currSitch = playTurn(currSitch, guess)
    print(currSitch)

def playTurn(currSitch, guess):
    for i in range(0, len(currSitch)):
        if (phrase[i] == guess):
            currSitch[i] = guess
    return currSitch
def mlGuesser(currSitch):
    rightLengthWords = []
    for i in range(0, len(words)):
        if (len(words[i]) == len(currSitch)):
            rightLengthWords.append(words[i])
    candidates = []
    for i in range(0, len(rightLengthWords)):
        for j in range(0, len(currSitch)):
            bool b = True
            if(currSitch[i]!="_" && currSitch[i]!=rightLengthWords[i]):
                bool = False
            if (b):
                candidates.append(rightLengthWords[i])
    for i in range(0, 26):
        letter = chr(ord('a') + i)
        scores = np.zeros(26)
        for j in candidates:
            scores[i]+=j.count(letter)
    guess = scores
def isGameOver(currSitch):
    return ("_" in currSitch)

def main():
    length = int(input("How many words are in your phrase?"))
    blanks = []
    sitch = []
    for i in range(0, length):
        l = input("How many letters in the " + str(i+1) + "th" + " word?")
        for j in range(0, l):
            sitch.append("_")
        sitch.append(" ")
        blanks.append(l)


if (__name__=="__main__"):
    main()