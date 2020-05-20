import os
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
import csv

#https://www.epcc.ed.ac.uk/blog/2016/08/23/mpi-performance-knl

def plot_execution_time(filename):
    proc_num  = []
    exec_time_sect = []
    exec_time_task = []
    exec_time_cube = []

    with open(filename, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=' ')
        for row in plots:
            proc_num.append(int(row[0]))
            exec_time_sect.append(float(row[1]))
            exec_time_task.append(float(row[2]))
            exec_time_cube.append(float(row[3]))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.27, 11.69))

    tickssize = 10
    labelsize = 12
    titlesize = 14
    suptlsize = 18

    fig.suptitle("Parallel program performance on cluster for \n" + filename, fontsize=suptlsize, fontweight='bold')

    params = dict(  lw              = 1.6, 
                    #color           = color,
                    alpha           = 0.75,
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

    ax1.set_xlabel(r'Threads', fontsize=labelsize)
    ax1.set_ylabel(r'Execution time (sec)', fontsize=labelsize)
    ax1.set_title (r'Dependence of the execution time on the number of threads (merge sort)', fontsize=titlesize)
    ax1.plot(proc_num, exec_time_sect, **params, label='parallel (sections)')
    ax1.plot(proc_num, exec_time_task, **params, label='parallel (tasks)')
    ax1.plot(proc_num, exec_time_cube, **params, label='parallel (hypercube)')

    ax2.set_xlabel(r'Threads', fontsize=labelsize)
    ax2.set_ylabel(r'Speedup', fontsize=labelsize)
    ax2.set_title (r'Dependence of the speedup time on the number of threds',    fontsize=titlesize)
    ax2.plot(proc_num, np.divide(exec_time_sect[0], exec_time_sect), **params, label='parallel (sections)')
    ax2.plot(proc_num, np.divide(exec_time_task[0], exec_time_task), **params, label='parallel (tasks)')
    ax2.plot(proc_num, np.divide(exec_time_cube[0], exec_time_cube), **params, label='parallel (hypercube)')

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