import matplotlib.pyplot as plt
import numpy as np
import csv
import os


def main():
    cwd = os.getcwd()
    path_parent = os.path.dirname(cwd)
    fileName = path_parent + "/diffDyn.csv"
    print(fileName)
    file = open(fileName)
    csvreader = csv.reader(file)


    header = []
    header = next(csvreader)
    print(header) 

    numColumns = 24

    

    rows = []
    for row in csvreader:
        rows.append(row)

    myData = np.zeros((len(rows), numColumns))

    # read csv row data into arrays of floats
    for i in range(numColumns - 1):
        for j in range(len(rows)):

            #print(rows[i][j])
            #print(j)
            dataElement = rows[j][i]

            #print(type(dataElement))
            try:
                myData[j][i] = float(dataElement)
            except ValueError:
                print("coudltn convert to float: i, j: " + str(i) + " " + str(j))

    # create a figure

    #ax1 = plt.subplot()

    # Figure for all box positions
    fig0, axs0 = plt.subplots(2, 2, constrained_layout = True, sharey = False)
    fig0.suptitle('Box positions', fontsize=16)
    fig0.set_figwidth(200)
    fig0.set_figheight(100)
    index = 0
    for nn, ax in enumerate(axs0.flat):

        subPlotTitle = "box " + str(index)
        ax.set_title(subPlotTitle, fontsize=10, loc='left')
        ax.plot(myData[:,(index * 3)], lw=2.5, c = 'b')
        ax.plot(myData[:,(index * 3) + 1], lw=2.5, c = 'tab:orange')
        index += 1

    # Figure for all box velocities
    fig1, axs1 = plt.subplots(2, 2, constrained_layout = True, sharey = False)
    fig1.suptitle('Box velocities', fontsize=16)
    fig1.set_figwidth(200)
    fig1.set_figheight(100)
    index = 0
    offset = 3
    for nn, ax in enumerate(axs1.flat):

        subPlotTitle = "Joint " + str(index)
        ax.set_title(subPlotTitle, fontsize=10, loc='left')
        ax.plot(myData[:,offset + (index * 3)], lw=2.5, c = 'b')
        ax.plot(myData[:,offset + (index * 3) + 1], lw=2.5, c = 'tab:orange')
        index += 1


    plt.show()


main()