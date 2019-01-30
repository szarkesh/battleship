import numpy as np
from collections import defaultdict
import numpy.random as random
import math
import matplotlib.pyplot as plt
import numpy as np
import pdb
from copy import deepcopy
#baselineGuessScores = np.zeros(shape=(10,10))
correctGuesses = np.zeros(shape=(10,10))
wrongGuesses = np.zeros(shape=(10,10))
noGrid = False
hitDict = dict()
missDict = dict()
map = dict()
entropy = []
for i in range(0, 100): # reset entropy
    entropy.append([])

currentGrid = np.zeros(shape=(10,10))
unblocked = np.zeros(shape=(10,10))
# for i in range(0, int(math.pow(3,8))):
#     hitDict[i]=0
#     missDict[i]=0
def createInput():
    grid = np.zeros(shape=(10,10))
    ships = [5,4,3,3,2]
    for i in range(0,len(ships)):
        isVertical = random.choice([True, False])
        tryAgain = True
        while (tryAgain==True):
            tryAgain = False
            if (isVertical):
                xVal = random.randint(0,10)
                yVal = random.randint(0,11-ships[i])
                for j in range(0, ships[i]):
                    if (grid[xVal][yVal+j]!=0):
                        tryAgain = True
                        break
                if(tryAgain==False):
                    for k in range(0, ships[i]):
                        grid[xVal][yVal+k]=i+1
            else:
                xVal = random.randint(0, 11-ships[i])
                yVal = random.randint(0, 10)
                for j in range(0, ships[i]):
                    if (grid[xVal+j][yVal]!=0):
                        tryAgain = True
                        break
                if(tryAgain==False):
                    for j in range(0,ships[i]):
                        grid[xVal+j][yVal] = i+1
    return grid
def makeGuess(verbose = True, guesser = "stochastic", interface = False):
    global unblocked
    global currentGrid
    unblocked = np.zeros(shape=(10,10))
    probableTuples = []
    for i in range(0, 10):
        for j in range(0, 10):
            probableTuples.append((i,j))
    random.shuffle(probableTuples)
    unblocked[5][5] = 1
    guessCounter = 0
    if (verbose):
        print(currentGrid)
    hits = 0
    if (interface == False):
        while (isGameOver()==False):
            #newGuess = randomGuesser(currGrid, unblocked)
            #newGuess = stochasticGuesser()
            if(guesser=="random"):
                newGuess = randomGuesser()
                p = (17-hits)/(100-guessCounter)
                ans = -p*math.log(p)*(100-guessCounter)
                entropy[guessCounter].append(ans)
            if (guesser=="stochastic"):
                newGuess = stochasticGuesser()
            if (guesser=="ml"):
                newGuess = machineLearningGuesser()
                probs = []
                ans = 0
                for i in range(0,10):
                    list = []
                    for j in range(0,10):
                        if (check(i,j)==-1):
                            layoutNums = situation(i,j)
                            num = map[layoutNums]
                            if (num not in hitDict or num not in missDict):
                                prob = 0.2
                            else:
                                prob = (hitDict[num]+0.01)/(missDict[num]+hitDict[num]+0.01)
                        else:
                            prob=0
                        probs.append(prob)
                normalizedProbs = []
                for i in range(0, len(probs)):
                    normalizedProbs.append(probs[i]*(17-hits)/sum(probs))
                #print(probs)
                #print(sum(normalizedProbs))
                for i in range(0, len(probs)):
                    if (probs[i]!=0):
                        ans+=-normalizedProbs[i]*math.log(normalizedProbs[i])
                entropy[guessCounter].append(ans)
                    #print(ans)
            if (guesser=="mlposition"):
                newGuess = machineLearningMultiplier()
            response = guess(newGuess[0], newGuess[1])
            if (response=="hit"):
                hits +=1
            guessCounter+=1
    else:
        guessCounter=0
        while(hits<17):
            newGuess = machineLearningGuesser()
            response = guess(newGuess[0], newGuess[1])
            guessCounter+=1
            if (response=="hit"):
                hits +=1
        print("Game. Over. You got REKT in just " + str(guessCounter) + " moves!")
        currentGrid = np.zeros(shape=(10, 10))
        for i in range(0, 10):
            for j in range(0, 10):
                currentGrid[i][j] -= 1
        makeGuess(False, "ml", interface=True)
    if (verbose):
        printGrid()
        print("Number of Guesses:" + str(guessCounter))
    return guessCounter

def check(i,j):
    if (i>9 or i<0 or j>9 or j<0):
        return 0
    if (noGrid):
        return currentGrid[i][j]
    if (unblocked[i][j]==0):
        return -1
    elif(currentGrid[i][j]==0):
        return 0
    else:
        return 1
def isGameOver():
    ans = True
    for i in range(0, len(currentGrid)):
        for j in range(0, len(currentGrid[0])):
            if (currentGrid[i][j]!=0 and unblocked[i][j]==0):
                return False
    return True
def printGrid():
    for i in range(0, len(currentGrid)):
        s = ""
        for j in range(0, len(currentGrid[0])):
            if (unblocked[i][j]==0):
                s = s + "."
            if(unblocked[i][j]==1):
                if (currentGrid[i][j]==0):
                    s = s + "0"
                else:
                    s = s + "1"
        print(s)

def guess(i,j):
    unblocked[i][j] = 1
    if (noGrid==False):
        if (currentGrid[i][j]!=0):
            #baselineGuessScores[i][j]+=0.05
            correctGuesses[i][j]+=1
            return "hit"
        else:
            wrongGuesses[i][j]+=1
            #baselineGuessScores[i][j]-=0.012
            return "miss"
    else:
        print("Is there a ship at (" + str(i) + "," + str(j) + ")?")
        ans = input()
        if (ans=="M"):
            currentGrid[i][j] = 0
            print(currentGrid)
            return "miss"
        elif (ans[0] == "S"):
            shipLength =  int(ans[1])
            #print(shipLength)
            currentGrid[i][j] = 1
            for k in range(i-(shipLength-1),i+1):
                print("k", k)
                b = True
                for z in range(k, k+shipLength):
                    #print(i,z, check(i,z))
                    if (check(i,z)<=0):
                        b = False
                        #print(i,z)
                print(b)
                if (b):
                    for z in range(k, k+shipLength):
                        #print("reset2")
                        currentGrid[i][z] = 0
            for k in range(j-(shipLength-1), j+1):
                b = True
                for z in range(k, k+shipLength):
                    if (check(z,j)<=0):
                        b=False
                        #print(i, z)
                print(b)
                if (b):
                    for z in range (k, k+shipLength):
                        #print("reset")
                        currentGrid[z][j] = 0
            print(currentGrid)
            return "hit"

            #currentGrid[i][j]
        elif(ans == "H"):
            currentGrid[i][j] = 1
            print(currentGrid)
            return "hit"
def randomGuesser():
    validGuess = False
    while (validGuess == False):
        x = random.randint(0, 10)
        y = random.randint(0,10)
        if (unblocked[x][y]==0):
            validGuess = True
    return x,y
def stochasticGuesser():
    probableTuples = []
    guessScores = np.zeros(shape=(10,10))
    for i in range(0, 10):
        for j in range(0,10):
            probableTuples.append((i,j))
    random.shuffle(probableTuples)
    for i in range(0, 10):
        for j in range(0, 10):
            if (check(i, j) == -1):
                if (i == 0 or i == 9 or j == 0 or j == 10):
                    guessScores[i][j] -= 0.1
                guessScores[i][j]+=1
                if (i < 9 and check(i+1,j) == 1):
                    guessScores[i][j]+=1
                    if (i<8 and check(i+2,j)==1):
                        guessScores[i][j]+=1
                if (i >0 and check(i -1, j) == 1):
                    guessScores[i][j] += 1
                    if (i> 1 and check(i-2, j) == 1):
                        guessScores[i][j] += 1
                if (j < 9 and check(i, j+1) == 1):
                    guessScores[i][j] += 1
                    if (j < 8 and check( i, j + 2) == 1):
                        guessScores[i][j] += 1
                if (j >0 and check(i, j -1) == 1):
                    guessScores[i][j] += 1
                    if (j >1 and check(i, j -2) == 1):
                        guessScores[i][j] += 1


                #if there's nothing around it subtract a bit
                if (i < 9 and check(i+1,j) == 0):
                    guessScores[i][j]-=0.6
                    if (i<8 and check(i+2,j)==0):
                        guessScores[i][j]-=0.3
                if (i >0 and check(i -1, j) == 0):
                    guessScores[i][j] -=0.6
                    if (i> 1 and check(i-2, j) == 0):
                        guessScores[i][j] -= 0.3
                if (j < 9 and check(i, j + 1) == 0):
                    guessScores[i][j] -= 0.6
                    if (j < 8 and check(i, j + 2) == 0):
                        guessScores[i][j] -= 0.3
                if (j >0 and check(i, j -1) == 0):
                    guessScores[i][j]-= 0.6
                    if (j >1 and check(i, j -2) == 0):
                        guessScores[i][j] -= 0.3

            else:
                guessScores[i][j]=-1000
    #print(guessScores)
    maxVal = -100000
    maxX = 0
    maxY = 0
    ties = []
    for i in range(0,10):
        for j in range(0,10):
            multiplier = (correctGuesses[i][j]+1)/(correctGuesses[i][j]+wrongGuesses[i][j]+1)
            #multiplier = 1
            #print(multiplier)
            biasedIndex = multiplier*guessScores[i][j]
            if (biasedIndex>maxVal):
                ties = [(i,j)]
                maxX = i
                maxY = j
                maxVal = biasedIndex
            if (biasedIndex==maxVal):
                ties.append((i,j))
    #print(len(ties))
    #printGrid(currGrid,unblocked)
    guess = ties[random.randint(0,len(ties))]
    if (maxVal==0):
        return 5,5
    #for i in range(0,10):
        #print(guessScores[i].tolist())
    #print("next")
    return guess[0],guess[1]
def machineLearningGuesser():
    global map
    guessScores = np.zeros(shape=(10,10))
    for i in range(0, 10):
        for j in range(0, 10):
            index = map[situation(i, j)]
            #print(index)
            if (index in hitDict):
                guessScores[i][j] = (hitDict[index] + 1) / (missDict[index] + hitDict[index] + 1)
            else:
                guessScores[i][j] = 1
    maxVal = -1000
    for i in range(0,10):
        for j in range(0,10):
            if (guessScores[i][j]>maxVal and unblocked[i][j]==0):
                ties = [(i,j)]
                maxX = i
                maxY = j
                maxVal = guessScores[i][j]
            if (guessScores[i][j]==maxVal and unblocked[i][j]==0):
                ties.append((i,j))
    #print(len(ties))
    #printGrid(currGrid,unblocked)
    #print(maxVal)
    guess = ties[random.randint(0,len(ties))]
    layoutNums = situation(guess[0], guess[1])
    num = map[layoutNums]
    if (currentGrid[guess[0]][guess[1]]==0):
        if (num in missDict):
            missDict[num]+=1
        else:
            missDict[num]=1
            hitDict[num]=0
    else:
        if(num in hitDict):
            hitDict[num]+=1
        else:
            hitDict[num]=1
            missDict[num]=0
    #print(guess[0],guess[1])
    return guess[0],guess[1]
def machineLearningMultiplier():
    guessScores = np.zeros(shape=(10, 10))
    for i in range(0, 10):
        for j in range(0, 10):
            index = map[situation(i, j)]
            #print(index)
            if (check(i,j)==-1):
                guessScores[i][j] = (hitDict[index]+0.3)/(missDict[index]+hitDict[index]+1)
    maxVal = -100000
    maxX = 0
    maxY = 0
    ties = []
    for i in range(0,10):
        for j in range(0,10):
            multiplier = (correctGuesses[i][j]+1)/(correctGuesses[i][j]+wrongGuesses[i][j]+1)
            #multiplier = 1
            #print(multiplier)
            biasedIndex = multiplier*guessScores[i][j]
            if (biasedIndex>maxVal):
                ties = [(i,j)]
                maxX = i
                maxY = j
                maxVal = biasedIndex
            if (biasedIndex==maxVal):
                ties.append((i,j))
    guess = ties[random.randint(0, len(ties))]
    if (maxVal == 0):
        return 5, 5
        # for i in range(0,10):
        # print(guessScores[i].tolist())
    # print("next")
    return guess[0], guess[1]
def situation(i,j):
    layoutNums = []
    ans = 0
    layoutNums.append(check(i+1,j))
    layoutNums.append(check(i+2, j))
    layoutNums.append(check(i, j + 1))
    layoutNums.append(check(i, j + 2))
    layoutNums.append(check(i-1, j))
    layoutNums.append(check(i-2, j))
    layoutNums.append(check(i, j-1))
    layoutNums.append(check(i, j-2))
    for i in range(0,8):
        layoutNums[i] = layoutNums[i]+1
    #print(layoutNums)
    #print("layout", ans)
    return tuple(layoutNums)

def createPossibilities():
    dictionary = dict()
    answers = []
    for i in range(0, int(math.pow(3,8))):
        list = []
        n=i
        for j in range(0, 8):
            list.append(n % 3)
            n = int(n/3)
        answers.append(list)
    #print(answers)
    for i in range(0, len(answers)):
        dictionary[tuple(answers[i])] = checkRotations(answers[i])
        #print(dictionary[tuple(answers[i])])
    return dictionary
def checkRotations(layoutNums):
    rotations = []
    rotations.append(layoutNums)
    rotations.append(layoutNums[6:] + layoutNums[:6]) # rotate 90
    rotations.append(layoutNums[4:] + layoutNums[:4]) # rotate 180
    rotations.append(layoutNums[2:] + layoutNums[:2]) # rotate 270
    flipped = layoutNums[:2] + layoutNums[6:] + layoutNums[2:4] + layoutNums[4:6]
    rotations.append(flipped)
    rotations.append(flipped[6:] + flipped[:6])
    rotations.append(flipped[4:] + flipped[:4])
    rotations.append(flipped[2:] + flipped[:2])
    answers = []
    for j in rotations:
        ans = 0
        for i in range(0, len(j)):
            ans += (j[i]) * math.pow(3, i)
        answers.append(ans)
    #print(min(answers))
    return min(answers)

def testBoard():
    a=np.zeros((10,10))
    b=np.zeros((10,10))
    c=np.zeros((10,10))
    d=np.zeros((10,10))
    for i in range(0,10000):
        currentGrid = createInput()
        if (currentGrid[4][6]>0):
            for j in range(0,10):
                for k in range(0,10):
                    if (currentGrid[j][k]>0):
                        a[j][k]+=1
                    else:
                        b[j][k]+=1
        else:
            for j in range(0,10):
                for k in range(0,10):
                    if (currentGrid[j][k]>0):
                        c[j][k]+=1
                    else:
                        d[j][k]+=1
    print(a)
    print(b)
    print(c)
    print(d)
    heatmap = []
    for j in range(0, 10):
        list = []
        for k in range(0,10):
            if (j!=4 or k!=6):
                sum = a[j][k] + b[j][k] + c[j][k] + d[j][k]
                probBoth = a[j][k]/sum
                probFirst = b[j][k]/sum
                probSecond = c[j][k]/sum
                probNone = d[j][k]/sum
                probA=(a[j][k]+b[j][k])/sum
                probB=(a[j][k]+c[j][k])/sum
                print(sum, probBoth, probA, probB, probNone, probFirst, probSecond)
                mutualInfo = probBoth*math.log(probBoth/(probA*probB)) + probFirst*math.log(probFirst/(probA*(1-probB))) \
                             + probSecond*math.log(probSecond/(probB*(1-probA))) + probNone*math.log(probNone/((1-probB)*(1-probA)))
                list.append(mutualInfo)
            else:
                list.append(0)
        heatmap.append(list)
    heatmap = np.asarray(heatmap)
    np.set_printoptions(precision=2)
    print(heatmap)
    plt.imshow(heatmap, cmap='hot', interpolation='nearest')
    plt.title("Mutual Information of Squares on Grid with Square at Position (4,6)", y=1.05)
    #plt.colorbar(heatmap)
    #plt.legend("Brighter color = higher M.I.")
    plt.xticks(np.arange(0, 10, 1.0))
    plt.yticks(np.arange(0, 10, 1.0))
    plt.savefig("sup.png")
    plt.show()
def randomEntropyChecker():
    global correctGuesses
    global wrongGuesses
    global currentGrid
    global entropy
    global map
    map = createPossibilities()
    for k in range(0,300):
        currentGrid = createInput()
        makeGuess(False, "ml")
    realans = []
    for i in range(0, len(entropy)):
        realans.append(sum(entropy[i])/len(entropy[0]))
    entropy = []


    for i in range(0, 100): # reset entropy
        entropy.append([])


    for k in range(0,1000): # time to test the random guesser
        currentGrid = createInput()
        makeGuess(False, "random")
    randomans = []
    for i in range(0, len(entropy)):
        randomans.append(sum(entropy[i]) / len(entropy[0]))
    print('hi')
    plt.plot(range(0,100), realans, label='Heuristic Algorithm')
    plt.plot(range(0,100), randomans,label='Random Guesser')
    plt.legend(loc='upper right')
    plt.title("Board Entropy of Random Guesser vs Board Entropy of Heuristic Algorithm", y=1.05)
    plt.xlabel("Moves")
    plt.ylabel("Board Entropy")
    plt.ylim([0, 35])
    plt.savefig("newentropy.png")
    print(realans)

def runforAiyer():
    global noGrid
    global currentGrid
    global map
    map = createPossibilities()
    for k in range(0, 20):
        #print(len(hitDict))
        turnsTaken = []
        for i in range(0, 100):
            currentGrid = createInput()
            turnsTaken.append(makeGuess(False, "ml"))
        # arr = []
        # for i in range(0,10):
        #     list = []
        #     for j in range(0,10):
        #         list.append(correctGuesses[i][j]/(wrongGuesses[i][j]+correctGuesses[i][j]))
        #     arr.append(list)
        print(np.mean(turnsTaken))
    currentGrid = np.zeros(shape=(10,10))
    noGrid = True
    for i in range(0,10):
        for j in range(0,10):
            currentGrid[i][j] -= 1
    makeGuess(False, interface = True)


def main():
    global correctGuesses
    global wrongGuesses
    global currentGrid
    global map
    map = createPossibilities()
    for k in range(0, 2):
        print(len(hitDict))
        turnsTaken = []
        arr = []
        for i in hitDict:
            arr.append((hitDict[i]+1)/(hitDict[i]+missDict[i]+1))
        print(arr)
        for i in range(0, 100):
            currentGrid = createInput()
            turnsTaken.append(makeGuess(False, "random"))
        arr = []
        for i in range(0,10):
            list = []
            for j in range(0,10):
                list.append(correctGuesses[i][j]/(wrongGuesses[i][j]+correctGuesses[i][j]))
            arr.append(list)
        print(np.mean(turnsTaken))
    correctGuesses = np.zeros(shape=(10,10))
    wrongGuesses = np.zeros(shape=(10,10))
        #print(np.array(arr))
    for k in range(0, 50):
        turnsTaken = []
        for i in range(0,50):
            currentGrid = createInput()
            turnsTaken.append(makeGuess(False, "mlposition"))
        print(np.mean(turnsTaken))

if(__name__ == "__main__"):
    runforAiyer()