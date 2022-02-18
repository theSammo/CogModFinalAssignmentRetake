import numpy as np
import pandas as pd
import string
from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import re
import logging 
# from scipy.special import softmax


def softmaxFunction(qvals, beta):
	sumqvals = 0
	pvals = []
	for qval in qvals:
		sumqvals += np.exp(beta*qval)
	for qval in qvals:
		pvals.append(np.exp(beta*qval)/sumqvals)
	return pvals


'''translates string to positions. 
if asletter, the second coordinate is not converted to a number'''
def translateToCoord(posID, asletter=True):
	splitCoord = re.findall('(\d+|[A-Za-z]+)',posID)
	firstCoord = int(splitCoord[0])
	if asletter:
		secondCoord = splitCoord[1]
	else:
		secondCoord = ord(splitCoord[1]) - 97 #convert letters to numbers 0 - 25
	return (firstCoord, secondCoord)

'''changes an index letter with an addition
for example a + 1 becomes b
used for movement'''
def changeLetter(letter, add):
	return chr(ord(letter) + add)

'''World class
contains all the variables necessary to build a world
a list of these is passed to the main to run different world setups efficiently'''
class World:
    def __init__(self, width, height, pitfalls, begin, end, iterations, rGenned=False, maxRuns=50000):
        self.width = width
        self.height = height
        self.pitfalls = pitfalls
        self.begin = begin
        self.end = end
        self.iterations = iterations
        self.rGenned = rGenned #is this world generated? 
        self.maxRuns = maxRuns #if yes, it might not be solvable, so maxRuns is used prevent infinite loops
        

'''the main which runs all the simulations
worlds is the list of World classes to create many different environments without
    having to edit everything manually
save=True makes the program save all its shown graphs into the folder where it is located'''
def main(worlds, save = False):#runs the program for every World passed
    dataToSave = {} #a dictionary for storing all collected data
    
    for key, world in worlds.items():
        logging.basicConfig(level=logging.INFO)
        dataToSave[key] = pd.DataFrame(index=["data"],columns=["average SARSA","median SARSA","average Q-Learning","median Q-Learning","pitfalls","stepsToPlot SARSA","stepsToPlot Q-Learning"])
        dataToSave[key].at["data","pitfalls"] = world.pitfalls
        '''initializing all variables'''
        nTrial = world.iterations
        rGenned = world.rGenned
        maxRuns = world.maxRuns
        alpha, beta, gammaQL, gammaSARSA, epsilon = 0.1, 3, 1, 1, 0.1
        actions = np.array(['up', 'left', 'down', 'right']) #possible actions in each state
        methods = ['e-greedy']#,'softmax', 'greedy']
        colors = ['red']#graph colors           ['green', 'yellow', 'red']
        # colorsMatched = dict(zip(methods, colors))
        s_0 = world.begin #start state
        s_terminal = world.end #final state (goal)
        xsize = world.width #width of the world
        ysize = world.height #height of the world
        potholes = world.pitfalls #all the holes you can fall into
        '''converting to coordinates
        "1a" would become (1, 0)'''
        for i, pothole in enumerate(potholes):
            potholes[i] = translateToCoord(pothole, True)
        columnLetters = list(string.ascii_lowercase)[:xsize] #letter labels for the graph
        lowest = ysize - 1 #lowest possible position
        rightmost = columnLetters[-1] #rightmost possible position
        # states= pd.DataFrame(np.empty([ysize,xsize],dtype=float),columns=columnLetters) #all possible states
        # for iY in range(ysize):
        #     for iX in columnLetters:
        #         states.loc[iY,iX] = iZ
        #         iZ +=1
        '''a dict to easily look up all state rewards
        look up rewards with Q3D[state][action]'''
        Q3D = {}
        fig, ax = plt.subplots()
        fig.set_size_inches(12,12)
        runs = {}
        totalRewards = []
        averageRewards = {}



        def move1step(state, direction):
            if (state[0],state[1]) in potholes:
                return translateToCoord(s_0)
            if (direction == 'up'):
                if (state[0] != 0):
                    next_state = (int(state[0])-1,state[1])
                else : next_state = state
            elif (direction == 'right'):
                if (state[1] != rightmost):
                    nextLetter = changeLetter(state[1], 1)
                    next_state = (state[0],nextLetter)
                else : next_state = state
            elif (direction == 'down'):
                if (state[0] != lowest):
                    next_state = (int(state[0])+1,state[1])
                else : next_state = state
            elif (direction == 'left'):
                if (state[1] != 'a'):
                    nextLetter = changeLetter(state[1], -1)
                    next_state = (state[0],nextLetter)
                else : next_state = state
            #apply rules so that you can derive the next state from the input variable(s)

            return(next_state) 

        def getReward(state):
            if f"{state[0]}{state[1]}" == s_terminal: #s_terminal:
                return 10
            elif (state[0],state[1]) in potholes:
                return -100
            else:
                return -1
        #the algorithm runs both learning algorithms and plots their results in the end
        for learningMethod in ['SARSA', 'Q-Learning']:
            if learningMethod == 'Q-Learning':
                
                gamma = gammaQL #separate gamma values for convenience
                '''I left the for loop below in from one of my earlier versions,
                since deleting it would make me have to do a lot of reworking of the code and
                this way it can be expanded with ease'''
                for method in methods: #eg q-learning
                    '''initializes runs[]
                    this dictionary of arrays keeps all run lengths for each configuration'''
                    runs[f"{method}{learningMethod}"] = []
                    averageRewards[f"{method}{learningMethod}"] = [] #total rewards of a run. name is from an earlier build
                    logging.info(f"{method} {learningMethod}")
                    for iY in range(ysize):
                        for iX in columnLetters:
                            Q3D[f"{iY}{iX}"] = [-1]*4 #initializes the Q-value table
                    for iT in range(nTrial): # loop for the different runs
                        if iT % 100 == 0:
                            logging.info(iT)                    
                        totalRewards = []
                        state = translateToCoord(s_0) # start in initial state as defined previously
                        finalState = translateToCoord(s_terminal)
                        run = 0
                        trialSteps = [] #saves all steps made by the program
                        while (state != finalState): # loop within one run
                            if rGenned and run >= maxRuns: #genned worlds might be unsolvable
                                break
                            run = run + 1
                            qvals = Q3D[f"{state[0]}{state[1]}"] # load qvalues for the current state
                            # select action using choice rules
                            if (method == 'softmax'):
                                pvals =  softmaxFunction(qvals, beta)# convert qvalues into probabilities
                                action = np.random.choice(actions,size = 1, p = pvals) # sample from  actions with specific probabilities          
                            elif (method == 'greedy'):
                                options = np.array(qvals == np.max(qvals)) #always selects the highest reward
                                action = np.random.choice(actions[options])
                            elif (method == 'e-greedy'):
                                options = np.array(qvals == np.max(qvals)) #selects highest reward with 1-epsilon probability
                                action = np.random.choice(actions[options])
                                if(np.random.random() < epsilon):
                                    action = np.random.choice(actions) #otherwise selects a random action


                            # interact with environment
                            # here we can use the function we defined earlier
                            next_state = move1step(state, action)
                            reward = getReward(next_state)
                            totalRewards.append(reward) #rewards of each step in a run
                            actionidx = (action==actions)
                            actIndex = [i for i, x in enumerate(actionidx) if x][0]
                            #updates the Q-value table
                            Q3D[f"{state[0]}{state[1]}"][actIndex] = Q3D[f"{state[0]}{state[1]}"][actIndex] + alpha*(reward + gamma * np.max(Q3D[f"{next_state[0]}{next_state[1]}"]) - Q3D[f"{state[0]}{state[1]}"][actIndex])
                            state = next_state #the new state becomes the old state
                            trialSteps.append(next_state)
                        averageRewards[f"{method}{learningMethod}"].append(totalRewards) 
                        runs[f"{method}{learningMethod}"].append(run)
                    '''to remove the randomness of the e-greedy method in the final evaluation,
                        I run a greedy algorithm over the Q-value table created by the
                        e-greedy algorithm to find what its preferred path is'''
                    if (method == 'e-greedy'):
                        logging.info("final")
                        state = translateToCoord(s_0) # start in initial state as defined previously
                        finalState = translateToCoord(s_terminal)
                        trialSteps = []
                        while (state != finalState): # loop within one run
                            if rGenned and run >= maxRuns:
                                break
                            qvals = Q3D[f"{state[0]}{state[1]}"] # load qvalues for the current state
                            # select action using choice rules
                            options = np.array(qvals == np.max(qvals))
                            action = np.random.choice(actions[options])
                            #intended path, so no randomization
                            '''if(np.random.random() < epsilon):
                                action = np.random.choice(actions)'''


                            # interact with environment
                            # here we can use the function we defined earlier
                            next_state = move1step(state, action)
                            reward = getReward(next_state)
                            
                            actionidx = (action==actions)
                            actIndex = [i for i, x in enumerate(actionidx) if x][0]
                            # update expectations using learing rules
                            # a few things are stil missing
                            Q3D[f"{state[0]}{state[1]}"][actIndex] = Q3D[f"{state[0]}{state[1]}"][actIndex] + alpha*(reward + gamma * np.max(Q3D[f"{next_state[0]}{next_state[1]}"]) - Q3D[f"{state[0]}{state[1]}"][actIndex])
                            # update variables
                            state = next_state #the new state becomes the old state
                            trialSteps.append(next_state)
                        startCoords = translateToCoord(s_0) #graph line start coordinate
                        stepsToPlot = [(ord(startCoords[1]) - 97, startCoords[0])] #adds this coordinate to the line plotter
                        for step in trialSteps: #for every step made, add a new position for the line to go through
                            stepsToPlot.append((ord(step[1]) - 97, step[0]))
                        logging.info(f"{method}: {stepsToPlot}")
                        dataToSave[key].at["data",f"stepsToPlot {learningMethod}"] = stepsToPlot
                        codes = [Path.MOVETO] + [Path.LINETO] * (len(stepsToPlot) - 1) #usage variables for the path graph
            
                        path = Path(stepsToPlot, codes) #creates the path graph of the steps

                        
                        patch = patches.PathPatch(path, facecolor='none', edgecolor='green', lw=2, label=f'e-greedy intended {learningMethod}')
                        ax.add_patch(patch) #adds the line to a graph
            if learningMethod == 'SARSA':
                #FOR NON-SARSA SPECIFIC FUNCTION COMMENTS, SEE "Q-Learning" ABOVE
                gamma = gammaSARSA
                for method in methods:
                    runs[f"{method}{learningMethod}"] = []
                    averageRewards[f"{method}{learningMethod}"] = []
                    logging.info(f"{method} {learningMethod}")
                    for iY in range(ysize):
                        for iX in columnLetters:
                            Q3D[f"{iY}{iX}"] = [0]*4
                    for iT in range(nTrial): # loop for the different runs
                        if iT % 100 == 0:
                            logging.info(iT)
                        state = translateToCoord(s_0) # start in initial state as defined previously
                        finalState = translateToCoord(s_terminal)
                        run = 0
                        trialSteps = []
                        totalRewards = []

                        qvals = Q3D[f"{state[0]}{state[1]}"] # load qvalues for the current state
                        options = np.array(qvals == np.max(qvals))
                        action = np.random.choice(actions[options]) #picks a first action
                        if(np.random.random() < epsilon):
                            action = np.random.choice(actions)
                        while (method == 'e-greedy' and state != finalState): # loop within one run
                            if rGenned and run >= maxRuns: #if it's randomly generated, it might be unsolvable
                                break
                            run = run + 1
                            sPrime = move1step(state,action) #sPrime is the next state based on 'action' and 'state'
                            qvals = Q3D[f"{sPrime[0]}{sPrime[1]}"] # load qvalues for that next state sPrime
                            # select the next action aPrime using choice rules
                            if (method == 'softmax'):
                                pvals =  softmaxFunction(qvals, beta)# convert qvalues into probabilities
                                aPrime = np.random.choice(actions,size = 1, p = pvals) # sample from  actions with specific probabilities          
                            elif (method == 'greedy'):
                                options = np.array(qvals == np.max(qvals))
                                aPrime = np.random.choice(actions[options])
                            elif (method == 'e-greedy'):
                                options = np.array(qvals == np.max(qvals))
                                aPrime = np.random.choice(actions[options])
                                if(np.random.random() < epsilon):
                                    aPrime = np.random.choice(actions)
                            reward = getReward(sPrime) #get the reward for the next state
                            totalRewards.append(reward)
                            actionidx = (action==actions)
                            actIndex = [i for i, x in enumerate(actionidx) if x][0]

                            actionPrimeidx = (aPrime==actions)
                            actPrimeIndex = [i for i, x in enumerate(actionPrimeidx) if x][0]
                            #update the Q-value table
                            Q3D[f"{state[0]}{state[1]}"][actIndex] = Q3D[f"{state[0]}{state[1]}"][actIndex] + alpha*(reward + gamma * Q3D[f"{sPrime[0]}{sPrime[1]}"][actPrimeIndex] - Q3D[f"{state[0]}{state[1]}"][actIndex])
                            state = sPrime
                            action = aPrime
                            trialSteps.append(sPrime)
                        averageRewards[f"{method}{learningMethod}"].append(totalRewards)
                        runs[f"{method}{learningMethod}"].append(run)

                    if (method == 'e-greedy'):
                        logging.info('final')
                        state = translateToCoord(s_0) # start in initial state as defined previously
                        finalState = translateToCoord(s_terminal)
                        trialSteps = []

                        qvals = Q3D[f"{state[0]}{state[1]}"] # load qvalues for the current state
                        options = np.array(qvals == np.max(qvals))
                        action = np.random.choice(actions[options])
                        while (state != finalState): # loop within one run
                            if rGenned and run >= maxRuns:
                                break
                            run = run + 1
                            sPrime = move1step(state,action)
                            qvals = Q3D[f"{sPrime[0]}{sPrime[1]}"] # load qvalues for the current state
                            # select action using choice rules
                            options = np.array(qvals == np.max(qvals))
                            aPrime = np.random.choice(actions[options])
                            reward = getReward(sPrime)
                            
                            actionidx = (action==actions)
                            actIndex = [i for i, x in enumerate(actionidx) if x][0]

                            actionPrimeidx = (aPrime==actions)
                            actPrimeIndex = [i for i, x in enumerate(actionPrimeidx) if x][0]
                            # update expectations using learing rules
                            # a few things are stil missing
                            Q3D[f"{state[0]}{state[1]}"][actIndex] = Q3D[f"{state[0]}{state[1]}"][actIndex] + alpha*(reward + gamma * Q3D[f"{sPrime[0]}{sPrime[1]}"][actPrimeIndex] - Q3D[f"{state[0]}{state[1]}"][actIndex])
                            # update variables
                            state = sPrime
                            action = aPrime
                            trialSteps.append(sPrime)
                        startCoords = translateToCoord(s_0)
                        stepsToPlot = [(ord(startCoords[1]) - 97+0.1, startCoords[0]+0.1)]
                        for step in trialSteps:
                            stepsToPlot.append((ord(step[1]) - 97+0.1, step[0]+0.1))
                        logging.info(f"{method}: {stepsToPlot}")
                        dataToSave[key].at["data",f"stepsToPlot {learningMethod}"] = stepsToPlot
                        codes = [Path.MOVETO] + [Path.LINETO] * (len(stepsToPlot) - 1)

                        path = Path(stepsToPlot, codes)

                        
                        patch = patches.PathPatch(path, facecolor='none', edgecolor='brown', lw=2, label=f'e-greedy intended {learningMethod}')
                        ax.add_patch(patch)
        #adding the start and end states to the graph
        '''WARNING: in my VS Code, I put a break point on the line coord = ... to
            prevent the program from softlocking. I don't know why it softlocks around here,
            but when I stopped VS Code and then let it continue again without changing
            anything, the code never softlocked again.
        This code draws colored boxes around the start and end states'''
        for state in ["begin", "end"]: 
            if state == "begin":
                coord = translateToCoord(s_0)
                numberCoord = (coord[0], ord(coord[1]) - 97)
            else:
                coord = translateToCoord(s_terminal)
                numberCoord = (coord[0], ord(coord[1]) - 97)
            potPoints = [ #4 points of the box and a "dump" point to close it
                (numberCoord[1] - 0.5, numberCoord[0] - 0.5),
                (numberCoord[1] - 0.5, numberCoord[0] + 0.5),
                (numberCoord[1] + 0.5, numberCoord[0] + 0.5),
                (numberCoord[1] + 0.5, numberCoord[0] - 0.5),
                (numberCoord[1] - 0.5, numberCoord[0] - 0.5)
            ]
            potCodes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY
            ]
            if state == "begin":
                patch = patches.PathPatch(Path(potPoints, potCodes), facecolor='blue', lw=2)
            else:
                patch = patches.PathPatch(Path(potPoints, potCodes), facecolor='orange', lw=2)
            ax.add_patch(patch)

        #adding the potholes to the graph
        '''this code draws black boxes around all of the 
            potholes in the world for easy visualization'''
        for pothole in potholes:
            numberCoord = (pothole[0], ord(pothole[1]) - 97)
            potPoints = [
                (numberCoord[1] - 0.5, numberCoord[0] - 0.5),
                (numberCoord[1] - 0.5, numberCoord[0] + 0.5),
                (numberCoord[1] + 0.5, numberCoord[0] + 0.5),
                (numberCoord[1] + 0.5, numberCoord[0] - 0.5),
                (numberCoord[1] - 0.5, numberCoord[0] - 0.5)
            ]
            potCodes = [
                Path.MOVETO,
                Path.LINETO,
                Path.LINETO,
                Path.LINETO,
                Path.CLOSEPOLY
            ]
            patch = patches.PathPatch(Path(potPoints, potCodes), facecolor='black', lw=2)
            ax.add_patch(patch)

        #setting graph parameters
        ax.set_xlim(-1, xsize)
        ax.set_ylim(-1, ysize)
        ax.set_xticks(np.arange(len(columnLetters)))
        ax.set_yticks(np.arange(ysize))
        ax.set_xticklabels(columnLetters)
        ax.set_yticklabels(np.arange(ysize))
        ax.set_aspect('equal')
        plt.legend()
        plt.gca().invert_yaxis()
        # plt.show()
        if save:
            fig.savefig(key)
        import statistics
        fig, ax = plt.subplots()  # Create a figure containing a single axis.
        fig2, ax2 = plt.subplots()
        xdata = range(nTrial)
        ax.set_title(f'Performance over time for decision methods')
        '''for each learning method x choice method, this creates a line
            in both graphs, with an automatically generated label'''
        for number, learningMethod in enumerate(['SARSA', 'Q-Learning']):
            colors = ["red", "green"]
            for method in methods:
                ydata = runs[f"{method}{learningMethod}"] #total runs for each trial
                from scipy.interpolate import make_interp_spline
                spl = make_interp_spline(xdata, ydata, k=3)
                ysmooth = spl(xdata)
                ax.plot(xdata, ysmooth, label = f"{method} {learningMethod}", color=colors[number], linewidth=.5, alpha=0.8)
                
                average = sum(runs[f"{method}{learningMethod}"])/len(runs[f"{method}{learningMethod}"])
                median = statistics.median(runs[f"{method}{learningMethod}"])
                print(f"{method} {learningMethod} average is " + str(average))
                print(f"{method} {learningMethod} median is " + str(median))
                dataToSave[key].at["data",f"average {learningMethod}"] = average
                dataToSave[key].at["data",f"median {learningMethod}"] = median
                averages = []
                '''this creates the total reward for a trial in a graph
                    as mentioned above, the name 'average' is from an earlier
                    build and should be ignored'''
                for i, runNumber in enumerate(range(nTrial)):
                    averages.append(sum(averageRewards[f"{method}{learningMethod}"][i]))
                spl = make_interp_spline(range(len(averages)),averages, k=3)
                smoothy = spl(range(len(averages)))
                ax2.plot(range(len(averages)), smoothy, label = f"{method} {learningMethod}", color=colors[number], linewidth=.5, alpha=0.8)
        ax.set_xlabel('Run')
        ax.set_ylabel('Number of moves')
        ax.set_xlim(0,250)
        ax.set_ylim(0,750)
        ax.legend()
        # fig.show()
        if save == True:
            fig.savefig(f"{key}_move")
        ax2.set_xlabel('Run')
        ax2.set_ylabel('Total reward')
        ax2.set_title("Total reward per run")
        ax2.set_xlim(0,250)
        ax2.set_ylim(-750,0)
        ax2.legend()
        # fig2.show()
        if save == True:
            fig2.savefig(f"{key}_total")
        print(f"{key} done")
    saveDF = pd.concat(dataToSave)
    if save:
        saveDF.to_excel(f'./allData.xlsx', index = True)






if __name__ == "__main__":
    # np.random.seed(42) #for reproducible results
    worlds = {} #a dict of all worlds

    #creates a handmade set of pitfalls for an interesting environments
    pitfalls = []
    pitfalls.append(f"3b")
    pitfalls.append(f"3c")
    pitfalls.append(f"3d")
    pitfalls.append(f"3e")
    pitfalls.append(f"3f")
    pitfalls.append(f"3g")
    pitfalls.append(f"3h")
    pitfalls.append(f"3i")
    pitfalls.append(f"3j")
    pitfalls.append(f"3k")
    #creates a world with (width, height, [pitfalls], startpos, endpos, max iterations)
    worlds['exp0'] = World(12, 4, pitfalls, '3a', '3l', 1000)



    pitfalls = []
    for height in range(1, 19):
        pitfalls.append(f"{height}d")
        if height % 3 == 0:
            pitfalls.append(f"{height}a")
        if height % 3 == 1:
            pitfalls.append(f"{height}c")
    worlds['exp1'] = World(6, 20, pitfalls, '0a', '19b', 1000)

    pitfalls = []
    for height in range(1, 8):
        pitfalls.append(f"{height}c")
    for height in range(1, 8):
        pitfalls.append(f"{height}k")
    for height in range(3):
        pitfalls.append(f"{height}g")
    for height in range(8, 10):
        pitfalls.append(f"{height}g")
    for width in range(2, 11):
        pitfalls.append(f"4{chr(width + 97)}")
    worlds['exp2'] = World(14, 10, pitfalls, '0a', '3l', 2000)

    '''creates 7 random worlds of size 15x15
        the start and beginning are always on the top and bottom respectively,
        but their x-position is random
        a random amount of pitfalls is created and scattered through the world
        they can overlap with each other without problems due to the way I 
        made this program'''
    for expNumber in range(3, 10):
        '''the sentence below generates a string of the form
            "0X" where X can be any letter between the 2nd and 14th letter
            of the alphabet like 0g or 0d
            this creates a starting position somewhere on the top row'''
        startpos = f"0{chr(np.random.randint(1,15) + 97)}"
        endpos = f"14{chr(np.random.randint(1,15) + 97)}"

        pitfalls = []
        amountOfPitfalls = 10 + np.random.randint(0, 16) #10 to 25 pitfalls
        for pitfall in range(amountOfPitfalls):
            newPitfall = f"{np.random.randint(1,15)}{chr(np.random.randint(1,15) + 97)}"
            #a new pitfall can't be the start or end tile
            while newPitfall == startpos or newPitfall == endpos:
                newPitfall = f"{np.random.randint(1,15)}{chr(np.random.randint(1,15) + 97)}"
            pitfalls.append(newPitfall)
        
        
        '''randomly generated worlds might not be solvable, so rGenned is set to True
            this means that after a certain amount of runs in an iteration, the program cuts
            off if the goal can't be reached'''
        worlds[f"exp{expNumber}"] = World(15, 15, pitfalls, startpos, endpos, 5000, True)
    
    '''the random worlds below are different in that the start and beginning
        are always in the same place, namely the top left and bottom right respectively'''
    for expNumber in range(10, 20):
        '''the start and end positions are always the same'''
        startpos = "0a"
        endpos = "14o"

        pitfalls = []
        amountOfPitfalls = 10 + np.random.randint(0, 16)
        for pitfall in range(amountOfPitfalls):
            newPitfall = f"{np.random.randint(1,15)}{chr(np.random.randint(1,15) + 97)}"
            while newPitfall == startpos or newPitfall == endpos:
                newPitfall = f"{np.random.randint(1,15)}{chr(np.random.randint(1,15) + 97)}"
            pitfalls.append(newPitfall)
        worlds[f"exp{expNumber}"] = World(15, 15, pitfalls, startpos, endpos, 2000, True)
    '''runs the main program over all generated worlds
        I recommend commenting out some of the above "worlds[]" expressions to
        ease the load on your PC or study each setup individually'''
    np.random.seed(42) #for reproducible results
    main(worlds, False) 