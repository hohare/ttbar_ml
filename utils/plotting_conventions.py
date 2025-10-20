import matplotlib.pyplot as plt
from cycler import cycler

colors = ["#5790fc", "#f89c20", "#e42536", "#964a8b", "#9c9ca1", "#7a21dd"]
colors10 = ["#3f90da","#ffa90e", "#bd1f01", "#94a4a2", "#832db6", "#a96b59", "#e76300", "#b9ac70","#717581","#92dadd"]

plt.rcParams.update({
    #"lines.markersize" : 1,
    'axes.prop_cycle': cycler(color=colors)
    })