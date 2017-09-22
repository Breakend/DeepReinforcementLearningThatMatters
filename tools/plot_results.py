import matplotlib.pyplot as plt
from scipy.ndimage.filters import uniform_filter1d
import time
import numpy as np
import pandas as pd
from itertools import cycle
import argparse

from numpy import genfromtxt
from numpy.random import choice

# Make fonts pretty
plt.rcParams['text.usetex'] = True

def multiple_plot(average_vals_list, std_dev_list, traj_list, other_labels, env_name, smoothing_window=5, no_show=False, ignore_std=False, climit=None, extra_lines=None):
    fig = plt.figure(figsize=(16, 8))
    colors = ["#1f77b4", "#ff7f0e", "#d62728", "#9467bd", "#2ca02c", "#8c564b", "#e377c2", "#bcbd22", "#17becf"]
    linestyles = ['solid', 'dashed', 'dashdot', 'dotted']
    color_index = 0
    ax = plt.subplot() # Defines ax variable by creating an empty plot
    offset = 1

    # Set the tick labels font
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
        label.set_fontname('Arial')
        label.set_fontsize(28)
    if traj_list is None:
        traj_list = [None]*len(average_vals_list)
    limit = climit
    index = 0
    for average_vals, std_dev, label, trajs in zip(average_vals_list, std_dev_list, other_labels[:len(average_vals_list)], traj_list):
        index += 1
        
        if climit is None:
            limit = len(average_vals)

        # If we don't want reward smoothing, set smoothing window to size 1
        rewards_smoothed_1 = uniform_filter1d(average_vals, size=smoothing_window)[:limit]
        std_smoothed_1 = uniform_filter1d(std_dev, size=smoothing_window)[:limit]
        rewards_smoothed_1 = rewards_smoothed_1[:limit]
        std_dev = std_dev[:limit]
       
        if trajs is None:
            # in this case, we just want to use algorithm iterations, so just take the number of things we have.
            trajs = list(range(len(rewards_smoothed_1)))
        else:
            plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
            ax.xaxis.get_offset_text().set_fontsize(20)

        fill_color = colors[color_index]
        color_index += 1

        cum_rwd_1, = plt.plot(trajs, rewards_smoothed_1, label=label, color=fill_color, ls=linestyles[color_index % len(linestyles)])
        offset += 3
        if not ignore_std:
            # uncomment this to use error bars
            #plt.errorbar(trajs[::25 + offset], rewards_smoothed_1[::25 + offset], yerr=std_smoothed_1[::25 + offset], linestyle='None', color=fill_color, capsize=5)
            plt.fill_between(trajs, rewards_smoothed_1 + std_smoothed_1,   rewards_smoothed_1 - std_smoothed_1, alpha=0.3, edgecolor=fill_color, facecolor=fill_color)

    if extra_lines:
        for lin in extra_lines:
            plt.plot(trajs, np.repeat(lin, len(rewards_smoothed_1)), linestyle='-.', color = colors[color_index], linewidth=2.5, label=other_labels[index])
            color_index += 1
            index += 1

    axis_font = {'fontname':'Arial', 'size':'32'}
    plt.legend(loc='lower right', prop={'size' : 16})
    plt.xlabel("Iterations", **axis_font)
    if traj_list is not None and traj_list[0] is not None:
        plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
        ax.xaxis.get_offset_text().set_fontsize(20)
        plt.xlabel("Timesteps", **axis_font)
    else:
        plt.xlabel("Iterations", **axis_font)
    plt.ylabel("Average Return", **axis_font)
    plt.title("%s"% env_name, **axis_font)

    if no_show:
        fig.savefig('%s.pdf' % env_name, dpi=fig.dpi, bbox_inches='tight')
    else:
        plt.show()

    return fig

parser = argparse.ArgumentParser()
parser.add_argument("paths_to_progress_csvs", nargs="+", help="All the csvs associated with the data (Need AverageReturn, StdReturn, and TimestepsSoFar columns)")
parser.add_argument("env_name", help= "This is just the title of the plot and the filename." )
parser.add_argument("--save", action="store_true")
parser.add_argument("--ignore_std", action="store_true")
parser.add_argument('--labels', nargs='+', help='List of labels to go along with the lines to plot', required=False)
parser.add_argument('--smoothing_window', default=1, type=int, help="Running average to smooth with, default is 1 (i.e. no smoothing)")
parser.add_argument('--limit', default=None, type=int)
parser.add_argument('--extra_lines', nargs="+", type=float, help="Any extra lines to add on the graph (such as other paper resulls)")

args = parser.parse_args()

avg_rets = []
std_dev_rets = []
trajs = []

# Read all csv's into arrays
for o in args.paths_to_progress_csvs:
    data = pd.read_csv(o)
    avg_ret = np.array(data["AverageReturn"]) # averge return across trials
    std_dev_ret = np.array(data["StdReturn"]) # standard error across trials
    if "total/steps" in data and trajs is not None:
        trajs.append(np.array(data["total/steps"]))
    elif "TimestepsSoFar" in data and trajs is not None:
        trajs.append(np.array(data["TimestepsSoFar"]))
    else:
        trajs = None
    avg_rets.append(avg_ret)
    std_dev_rets.append(std_dev_ret)

# call plotting script
multiple_plot(avg_rets, std_dev_rets, trajs, args.labels, args.env_name, smoothing_window=args.smoothing_window, no_show=args.save, ignore_std=args.ignore_std, climit=args.limit, extra_lines=args.extra_lines)
