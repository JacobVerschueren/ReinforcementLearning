import numpy as np
import matplotlib.pyplot as plt

class MakeGraph:

    def __init__(self, title, data, ylable, xlable = 'episodes'):
        self.title = title
        self.data = data
        self.ylable = ylable
        self.xlable = xlable

    def drawGraph(self):

        t = np.arange(0.0, len(self.data))
        fig, ax = plt.subplots()
        ax.plot(t, self.data)

        ax.set(xlabel=self.xlable, ylabel=self.ylable,
               title=self.title)
        ax.grid()

        plt.show()