
import matplotlib.pyplot as plt # for drawing graphs
from pybbn.graph.jointree import JoinTree
from bnmetamodel_gh import *
import copy

###Create plot for probability vs bin ranges 
class CreatePlot:

    def plotPDs (self, maintitle, xlabel, ylabel, displayplt = False, **kwargs ): # plots the probability distributions


        # code to automatically set the number of columns and rows and dimensions of the figure
        jt = JoinTree()
        n_totalplots = len(self.node)
        n_totalplots = jt.nodes

        if n_totalplots <= 4:
            n_cols = n_totalplots
            n_rows = 1

        else: 
            n_cols = 4
            n_rows = n_totalplots % 4
        if n_rows == 0: n_rows = n_totalplots/4

        #generate the probability distributions for the prior distributions
        binRanges = self.BNdata.binRanges
        priorPDs = {}


        #min = binRanges[0][0]
        #max = binRanges[len(binRanges)][1]
        #binedges = bins(max, min, len(binRanges))
    
        bincounts = self.BNdata.bincountsDict

        for varName in bincounts:
            total = sum(sum(x) for x in bincounts[varName])
            priors = []
            for count in bincounts[varName]:
                priors.append(float(count[0])/float(total))

            priorPDs[varName] = priors

        #for varName in binRanges: draw_barchartpd(binRanges[varName],priorPDs[varName])

        # instantiate a figure as a placaholder for each distribution (axes)
        fig = plt.figure(figsize=((200 * n_cols) / 96, (200 * n_rows) / 96), dpi=96, facecolor='white')
        fig.suptitle(maintitle, fontsize=8) # title

        #sort evidence distributions to be plotted first
        #nodessorted = []

        #copy node names into new list
        nodessorted = copy.copy(self.nodes)

        # evidence
        evidenceVars = []
        if 'evidence' in kwargs:
            evidenceVars = kwargs['evidence']

            #sort evidence variables to be in the beginning of the list
            for index, var in enumerate (evidenceVars):
                nodessorted.insert(index, nodessorted.pop(nodessorted.index(evidenceVars[index])))

        i = 0

        for varName in nodessorted:
            # print df
            ax = fig.add_subplot(n_rows, n_cols, i + 1)
            ax.set_axis_bgcolor("whitesmoke")

            xticksv = []
            binwidths = []
            edge = []

            for index, range in enumerate(binRanges[varName]):
                edge.append(range[0])
                binwidths.append(range[1] - range[0])
                xticksv.append(((range[1] - range[0]) / 2) + range[0])
                if index == len(binRanges[varName]) - 1: edge.append(range[1])

            #df[var_name].hist(bins=binwidths[var_name],ax=ax)
            #plot the priors
            ax.bar(xticksv, priorPDs[varName], align='center', width=binwidths, color='black', alpha=0.2, linewidth=0.2)

            #filter out evidence and query to color the bars accordingly (evidence-green, query-red)
            if 'posteriorPD' in kwargs:


                if len(kwargs['posteriorPD'][varName]) > 1:
                    if varName in evidenceVars:
                        ax.bar(xticksv, kwargs['posteriorPD'][varName], align='center', width=binwidths, color='green', alpha=0.2, linewidth=0.2)

                    else:
                        ax.bar(xticksv, kwargs['posteriorPD'][varName], align='center', width=binwidths, color='red', alpha=0.2, linewidth=0.2)

            ##: fix xticks .... not plotting all
            #plt.xlim(edge[0], max(edge))
            plt.xticks([round(e, 4) for e in edge], rotation='vertical')
            plt.ylim(0, 1)
            #plt.show()

            for spine in ax.spines:
                ax.spines[spine].set_linewidth(0)

            ax.grid(color='0.2', linestyle=':', linewidth=0.1, dash_capstyle='round')
            # ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))
            ax.set_title(varName, fontweight="bold", size=6)
            ax.set_ylabel(ylabel, fontsize=7)  # Y label
            ax.set_xlabel(xlabel, fontsize=7)  # X label
            ax.xaxis.set_tick_params(labelsize=6, length =0)
            ax.yaxis.set_tick_params(labelsize=6, length = 0)


            # ax.grid(False)
            #if 'xlim' in kwargs:
            #    ax.set_xlim(kwargs['xlim'][0], kwargs['xlim'][1])

            i += 1

        fig.tight_layout()  # Improves appearance a bit.
        fig.subplots_adjust(top=0.85)  # white spacing between plots and title
        # if you want to set backgrond of figure to transpearaent do it here. Use facecolor='none' as argument in savefig ()

        if displayplt == True: plt.show()