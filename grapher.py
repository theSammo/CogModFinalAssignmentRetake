import pandas as pd
import numpy as np
import ast
from matplotlib import pyplot as plt
from matplotlib.path import Path
import matplotlib.patches as patches
import string

def unpackPitfalls(pitfalls):
    newPitfalls = []
    for pitfall in pitfalls:
        newPitfalls.append((pitfall[0], ord(pitfall[1]) - 97))
    return newPitfalls

def main(hasSaved):
    worldsizes = [(12, 4), (6, 20), (14, 10)]
    for i in range(17):
        worldsizes.append((15, 15))
    allData = pd.read_excel("allData.xlsx", index_col=[0,1])
    experimentAmount = allData.shape[0]
    allPathsSARSA = []
    allPathsQLearning = []
    pitfalls = []
    for i in range(experimentAmount):
        allPathsSARSA.append(list(ast.literal_eval(allData.loc[f"exp{i}"].loc["data"].loc["stepsToPlot SARSA"])))
        allPathsQLearning.append(list(ast.literal_eval(allData.loc[f"exp{i}"].loc["data"].loc["stepsToPlot Q-Learning"])))
        pitfalls.append(list(ast.literal_eval(allData.loc[f"exp{i}"].loc["data"].loc["pitfalls"])))
    intendedPaths = []
    fastestPaths = []
    if hasSaved:
        for i in range(experimentAmount):
            intendedPaths.append(list(ast.literal_eval(pd.read_excel("./Final Graphs/allPaths.xlsx").at[0, f"intended {i}"])))
            fastestPaths.append(list(ast.literal_eval(pd.read_excel("./Final Graphs/allPaths.xlsx").at[0, f"fastest {i}"])))
    else:
        pathsToAdd = {}
        for path in allPathsSARSA:
            intendedPaths.append([(x+0.1,y+0.1) for (x,y) in path])
        for path in allPathsQLearning:
            fastestPaths.append([(x-0.1,y-0.1) for (x,y) in path])


    translatedPitfalls = []
    for set in pitfalls:
        translatedPitfalls.append(unpackPitfalls(set))
    startPos = []
    endPos = []
    for i in range(experimentAmount):
        startPos.append(allPathsQLearning[i][0])
        endPos.append(allPathsQLearning[i][-1])

    for number, world in enumerate(allPathsSARSA):
        if not hasSaved:
            pathsToAdd[f"intended {number}"] = [intendedPaths[number]]
            pathsToAdd[f"fastest {number}"] = [fastestPaths[number]]
        fig, ax = plt.subplots()
        fig.set_size_inches(12,12)
        codes = [Path.MOVETO] + [Path.LINETO] * (len(allPathsSARSA[number]) - 1) #usage variables for the path graph
        path = Path(allPathsSARSA[number], codes) #creates the path graph of the steps
        patch = patches.PathPatch(path, facecolor='none', edgecolor='brown', lw=2, label='e-greedy intended SARSA')
        ax.add_patch(patch) #adds the line to a graph

        codes = [Path.MOVETO] + [Path.LINETO] * (len(allPathsQLearning[number]) - 1) #usage variables for the path graph
        path = Path(allPathsQLearning[number], codes) #creates the path graph of the steps
        patch = patches.PathPatch(path, facecolor='none', edgecolor='green', lw=2, label='e-greedy intended Q-Learning')
        ax.add_patch(patch) #adds the line to a graph

        codes = [Path.MOVETO] + [Path.LINETO] * (len(intendedPaths[number]) - 1) #usage variables for the path graph
        path = Path(intendedPaths[number], codes) #creates the path graph of the steps
        patch = patches.PathPatch(path, facecolor='none', edgecolor='pink', lw=2, label='least dangerous')
        ax.add_patch(patch) #adds the line to a graph

        codes = [Path.MOVETO] + [Path.LINETO] * (len(fastestPaths[number]) - 1) #usage variables for the path graph
        path = Path(fastestPaths[number], codes) #creates the path graph of the steps
        patch = patches.PathPatch(path, facecolor='none', edgecolor='blue', lw=2, label='fastest')
        ax.add_patch(patch) #adds the line to a graph

        '''This code draws colored boxes around the start and end states'''
        for state in ["begin", "end"]: 
            if state == "begin":
                coord = startPos[number]
                numberCoord = (coord[0], coord[1])
            else:
                coord = endPos[number]
                numberCoord = (coord[0], coord[1])
            potPoints = [ #4 points of the box and a "dump" point to close it
                (numberCoord[0] - 0.5, numberCoord[1] - 0.5),
                (numberCoord[0] - 0.5, numberCoord[1] + 0.5),
                (numberCoord[0] + 0.5, numberCoord[1] + 0.5),
                (numberCoord[0] + 0.5, numberCoord[1] - 0.5),
                (numberCoord[0] - 0.5, numberCoord[1] - 0.5)
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
        for pitfall in translatedPitfalls[number]:
            numberCoord = (pitfall[0], pitfall[1])
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
        columnLetters = list(string.ascii_lowercase)[:worldsizes[number][0]] #letter labels for the graph
        #setting graph parameters
        ax.set_xlim(-1, worldsizes[number][0])
        ax.set_ylim(-1, worldsizes[number][1])
        ax.set_xticks(np.arange(len(columnLetters)))
        ax.set_yticks(np.arange(worldsizes[number][1]))
        ax.set_xticklabels(columnLetters)
        ax.set_yticklabels(np.arange(worldsizes[number][1]))
        ax.set_aspect('equal')
        plt.legend()
        plt.gca().invert_yaxis()
        # plt.show()
        print(number)
        fig.savefig(f"./Final Graphs/endGraph{number}")
    if not hasSaved:
        newPaths = pd.DataFrame.from_dict(pathsToAdd)
        newPaths.to_excel("./Final Graphs/allPaths.xlsx", index=True)


def computeStats():
    minimumPitfalls = [0 ,3 , 0 , 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0]
    SARSAPitfalls = [0 ,3 , 0 , 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 1, 1, 0, 0, 3, 0, 0]
    QLearningPitfalls = [10,12, 11, 1, 1, 3, 3, 8, 1, 3, 2, 1, 3, 4, 4, 5, 3, 5, 5, 3]
    SDifference = [s - m for s,m in zip(SARSAPitfalls, minimumPitfalls)]
    QDifference = [s - m for s,m in zip(QLearningPitfalls, minimumPitfalls)]
    from statistics import mean
    SMeanDifference = mean(abs(x - y) for x,y in zip(SARSAPitfalls, minimumPitfalls))
    QMeanDifference = mean(abs(x - y) for x,y in zip(QLearningPitfalls, minimumPitfalls))
    SStd = np.std([abs(x - y) for x,y in zip(SARSAPitfalls, minimumPitfalls)])
    QStd = np.std([abs(x - y) for x,y in zip(QLearningPitfalls, minimumPitfalls)])
    print(f"SARSA unnecessary pitfalls per experiment: {SDifference}")
    print(f"SARSA mean unnecessary pitfalls: {SMeanDifference}")
    print(f"SARSA Standard Deviation: {SStd}")
    print(f"Q-Learning unnecessary pitfalls per experiment: {QDifference}")
    print(f"Q-Learning mean unnecessary pitfalls: {QMeanDifference}")
    print(f"Q-Learning Standard Deviation: {QStd}")
    allData = pd.read_excel("allData.xlsx", index_col=[0,1])
    intendedPaths = []
    fastestPaths = []
    allPathsSARSA = []
    allPathsQLearning = []
    for i in range(20):
        intendedPaths.append(list(ast.literal_eval(pd.read_excel("./Final Graphs/allPaths.xlsx").at[0, f"intended {i}"])))
        fastestPaths.append(list(ast.literal_eval(pd.read_excel("./Final Graphs/allPaths.xlsx").at[0, f"fastest {i}"])))
        allPathsSARSA.append(list(ast.literal_eval(allData.loc[f"exp{i}"].loc["data"].loc["stepsToPlot SARSA"])))
        allPathsQLearning.append(list(ast.literal_eval(allData.loc[f"exp{i}"].loc["data"].loc["stepsToPlot Q-Learning"])))
    SARSALengths = [len(path) - 1 for path in allPathsSARSA]
    QLearningLengths = [len(path) - 1 for path in allPathsQLearning]
    print(f"Minimum steps per experiment: {QLearningLengths}")
    print(f"SARSA steps per experiment: {SARSALengths}")
    SARSAExtraSteps = [abs(x - y) for x,y in zip(SARSALengths, QLearningLengths)]
    print(f"SARSA steps over minimum per experiment: {SARSAExtraSteps}")
    SMeanExtra = mean(SARSAExtraSteps)
    SStdExtra = np.std(SARSAExtraSteps)
    print(f"SARSA mean unnecessary steps: {SMeanExtra}")
    print(f"SARSA Standard Deviation: {SStdExtra}")
    print(f"Q-Learning steps per experiment: {QLearningLengths}")
    QLearningExtraSteps = [abs(x - y) for x,y in zip(QLearningLengths, QLearningLengths)]
    print(f"Q-Learning steps per experiment: {QLearningExtraSteps}")
    QMeanExtra = mean(QLearningExtraSteps)
    QStdExtra = np.std(QLearningExtraSteps)
    print(f"Q-Learning mean unnecessary steps: {QMeanExtra}")
    print(f"Q-Learning Standard Deviation: {QStdExtra}")

if __name__ == "__main__":
    main(True)
    computeStats()