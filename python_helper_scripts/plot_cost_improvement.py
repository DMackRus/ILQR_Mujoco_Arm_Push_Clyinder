import matplotlib.pyplot as plt
import numpy as np
import csv
import os



def main():
    cwd = os.getcwd()
    path_parent = os.path.dirname(cwd)
    fileName = path_parent + "/costs.csv"
    print(fileName)
    file = open(fileName)
    csvreader = csv.reader(file)

    header = []
    header = next(csvreader)
    print(header) 

    numColumns = len(header)
    print(numColumns)

    

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

    plt.plot(myData[:, 0], lw=2.5, c = 'b')
    plt.plot(myData[:, 1], lw=2.5, c = 'g')
    plt.plot(myData[:, 2], lw=2.5, c = 'r')
    plt.plot(myData[:, 3], lw=2.5, c = 'c')
    plt.plot(myData[:, 4], lw=2.5, c = 'm')
    #plt.plot(myData[:, 5], lw=2.5, c = 'y')
    #plt.plot(myData[:, 6], lw=2.5, c = 'tab:orange')
    #plt.plot(myData[:, 7], lw=2.5, c = 'tab:purple')

    # Figure for all joint positions
    # fig0, axs0 = plt.subplots(1, 1, constrained_layout = True, sharey = False)
    # fig0.suptitle('Costs over iterations (1 -> 7)', fontsize=16)
    # fig0.set_figwidth(200)
    # fig0.set_figheight(100)
    # index = 0
    # for nn, ax in enumerate(axs0.flat):
    #     if(index > 6):
    #         break
    #     subPlotTitle = "Joint " + str(index)
    #     ax.set_title(subPlotTitle, fontsize=10, loc='left')
    #     ax.plot(myData[:,(index * 3)], lw=2.5, c = 'b')
    #     ax.plot(myData[:,(index * 3) + 1], lw=2.5, c = 'tab:orange')
    #     index += 1



    plt.show()


main()