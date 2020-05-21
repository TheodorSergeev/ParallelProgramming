import os
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import csv

def plot_execution_time(filename):
    proc_num  = []
    ghost_num = []
    exec_time_synch = []
    exec_time_asynch = []

    with open(filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=' ')
        for row in plots:
            proc_num.append(int(row[0]))
            ghost_num.append(int(row[1]))
            exec_time_synch.append(float(row[2]))
            exec_time_asynch.append(float(row[3]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 15))

    tickssize = 10
    labelsize = 12
    titlesize = 14
    suptlsize = 18

    fig.suptitle("Parallel program performance on cluster for \n" + filename, fontsize=suptlsize, fontweight='bold')

    params = dict(  lw              = 1.8, 
                    #color           = color,
                    alpha           = 0.5,
                    markersize      = 4,
                    marker          = 'o',
                    markevery       = 1,
                    markerfacecolor = 'white',
                    #markeredgecolor = color,
                    markeredgewidth = 1.2) 

    plt.subplots_adjust(top=0.9, hspace=0.4)

    ax1.tick_params(direction='inout', length=6, width=1.5, colors='black')

    ax1.grid(linestyle=':')
    ax2.grid(linestyle=':')

    ax1.set_xticks(proc_num)
    ax2.set_xticks(proc_num)
    #ax1.set_xticklabels(msg_size, rotation=40, fontsize=tickssize)
    #ax2.set_xticklabels(msg_size, rotation=40, fontsize=tickssize)

    ax1.set_xlabel(r'Processes number', fontsize=labelsize)
    ax1.set_ylabel(r'Execution time (sec)', fontsize=labelsize)
    ax1.set_title (r'Dependence of the matrix-vector multiplication time on the number of processes', fontsize=titlesize)
    ax1.plot(proc_num, exec_time_synch,  **params, label='Synch')
    ax1.plot(proc_num, exec_time_asynch, **params, label='Asynch')

    ax2.set_xlabel(r'Processes number', fontsize=labelsize)
    ax2.set_ylabel(r'Speedup',    fontsize=labelsize)
    ax2.set_title (r'Dependence of the speedup on the number of processes',    fontsize=titlesize)
    ax2.plot(proc_num, np.true_divide(exec_time_synch[0], exec_time_synch), **params, label='Synch')
    ax2.plot(proc_num, np.true_divide(exec_time_asynch[0], exec_time_asynch), **params, label='Asynch')

    ax1.legend()
    ax2.legend()
    
    plt.savefig(filename + "_plot.pdf", format='pdf')
    #plt.show()


dir_name = "."

for filename in os.listdir(dir_name):
    if filename.endswith(".txt"): 
        plot_execution_time(dir_name + "/" + filename)
        continue
    else:
        continue