import matplotlib.pyplot as plt # for drawing graphs
from pybbn.graph.jointree import JoinTree

###Create plot for probability vs bin ranges 
class CreatePlot2:
    def plotPDs (self, maintitle, xlabel, ylabel, displayplt = False, **kwargs ): # plots the probability distributions
        
        # code to automatically set the number of columns and rows and dimensions of the figure
        jt = JoinTree()
        n_totalplots = len(self.nodes)
        n_totalplots = jt.nodes

        if n_totalplots <= 4:
            n_cols = n_totalplots
            n_rows = 1

        else: 
            n_cols = 4
            n_rows = n_totalplots % 4
        if n_rows == 0: n_rows = n_totalplots/4

        # instantiate a figure as a placaholder for each distribution (axes)
        fig = plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96, facecolor='white')
        fig.suptitle(maintitle, fontsize=8) # title
        
        
        ax = fig.add_subplot( i + 1)
        ax.set_axis_bgcolor("whitesmoke")

        #plot the priors
        ax.bar(xticksv, priorPDs[varName], align='center', width=binwidths, color='black', alpha=0.2, linewidth=0.2)

        ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round')
        # ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
        ax.set_title(varName, fontweight="bold", size=6)
        ax.set_ylabel(ylabel, fontsize=7)  # Y label
        ax.set_xlabel(xlabel, fontsize=7)  # X label
        ax.xaxis.set_tick_params(labelsize=6, length =0)
        ax.yaxis.set_tick_params(labelsize=6, length = 0)

        fig.tight_layout()  # Improves appearance a bit.
        fig.subplots_adjust(top=0.85)  # white spacing between plots and title
        # if you want to set backgrond of figure to transpearaent do it here. Use facecolor='none' as argument in savefig ()

        if displayplt == True: plt.show()

