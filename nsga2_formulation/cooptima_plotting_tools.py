__author__ = 'rgrout'

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.path import Path
from matplotlib.spines import Spine
import matplotlib.ticker as ticker

# Follow pattern from matplotlib example
def radar_factory(num_vars, frame='circle'):

    # Calc evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
    # Rotate theta so first axis at the top
    theta += np.pi/2

    def draw_poly_patch(self):
        verts = unit_poly_verts(theta)
        return plt.Polygon(verts, closed=True, edgecolor='k')

    def draw_circle_patch(self):
        return plt.Circle((0.5,0.5),0.5)

    patch_dict = {'polygon':draw_poly_patch, 'circle':draw_circle_patch}
    if frame not in patch_dict:
        raise ValueError('unknown value for `frame`: %s' % frame)

    class RadarAxes(PolarAxes):
        name = 'radar'
        RESOLUTION = 1
        draw_patch = patch_dict[frame]

        def fill(self, *args, **kwargs):
             """Override fill so that line is closed by default"""
             closed = kwargs.pop('closed', True)
             return super(RadarAxes, self).fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super(RadarAxes, self).plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.concatenate((x, [x[0]]))
                y = np.concatenate((y, [y[0]]))
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            return self.draw_patch()

        def _gen_axes_spines(self):
            if frame == 'circle':
                return PolarAxes._gen_axes_spines(self)
            # The following is a hack to get the spines (i.e. the axes frame)
            # to draw correctly for a polygon frame.

            # spine_type must be 'left', 'right', 'top', 'bottom', or `circle`.
            spine_type = 'circle'
            verts = unit_poly_verts(theta)
            # close off polygon by repeating first vertex
            verts.append(verts[0])
            path = Path(verts)

            spine = Spine(self, spine_type, path)
            spine.set_transform(self.transAxes)
            return {'polar': spine}

    register_projection(RadarAxes)
    return theta
        
def unit_poly_verts(theta):
    """Return vertices of polygon for subplot axes.

    This polygon is circumscribed by a unit circle centered at (0.5, 0.5)
    """
    x0, y0, r = [0.5] * 3
    verts = [(r*np.cos(t) + x0, r*np.sin(t) + y0) for t in theta]
    return verts
        



# Plot the composition using names from propDB as axis label on a radar plot
def plot_comp_radar(propDB, comp, title='Composition', savefile=None):
    N = len(comp)
    theta = radar_factory(N, frame='circle')

    spoke_lab = []
    spoke_vals = []

    for k in propDB.keys():
        spoke_lab.append(propDB[k]['NAME'])
        spoke_vals.append(comp[k])

    fig = plt.figure(figsize=(9,9))

    ax = fig.add_subplot(1,1,1,projection='radar')

    ax.set_title(title, weight='bold', size='medium',
                 position=(0.5,1.1),
                 horizontalalignment='center', verticalalignment='center')
    ax.plot(theta, spoke_vals)
    ax.fill(theta, spoke_vals, alpha=0.25)
    ax.set_varlabels(spoke_lab)

    if (savefile):
        plt.savefig(savefile+".pdf", form='pdf')
    else:
        plt.show()
    plt.close()
    


def plot_prop_parallel(proplist):
# Starting from http://stackoverflow.com/questions/8230638/parallel-coordinates-plot-in-matplotlib
    dims = len(proplist[0])
    x = range(dims)

    print proplist[0]

    fig, axs = plt.subplots(1,dims-1, sharey=False)
    style = ['r-']*len(proplist)

    min_per_prop = {}
    max_per_prop = {}
    range_per_prop = {}
    for p in proplist[0]:
        min_per_prop[p] = 1.0e20
        max_per_prop[p] = -1.0e20
        range_per_prop[p] = 0.0
    for prop in proplist:
#        print "propset: ", prop
        for p,v in prop.iteritems():
#            print "prop", p, " = ", v
            mn = min(min_per_prop[p], v)
            mx = max(max_per_prop[p], v)
            r = float(mx-mn)
            min_per_prop[p] = mn
            max_per_prop[p] = mx
            range_per_prop[p] = r


#Normalize
    nondim_vals = {}
    # fix order of properties
    prop_order = proplist[0].keys()
    nd_list = []

    for prop in proplist:
        nd_props= []
        for p in prop_order:
            print "normalizing: ", prop[p], " by min / range: ", min_per_prop[p], range_per_prop[p]
            if np.abs(range_per_prop[p]) > 0.0:
                normval = ( (prop[p] - min_per_prop[p])/range_per_prop[p] )
                print "norm val = ", normval
                nd_props.append( normval)
            else:
                nd_props.append( 0.5 )
        nd_list.append(nd_props)



# Print on each axis line between normalized property values

    for i, ax in enumerate(axs):
        for prop_index, ndval in enumerate(nd_list):
            ax.plot(x, ndval, style[prop_index])
            ax.set_xlim([x[i], x[i+1]])
    
# Clean up the x lables
    i = 0
    for ax, xx in zip (axs, x[:-1]):
        ax.xaxis.set_major_locator(ticker.FixedLocator([xx]))
        ax.set_xticklabels([prop_order[i]])
        i += 1
    axs[-1].xaxis.set_major_locator(ticker.FixedLocator([x[-2],x[-1]]))
    axs[-1].set_xticklabels(prop_order[-2:])

# Clean up the y lables
    for ax, p in zip(axs, prop_order):
        ax.yaxis.set_ticks([0,1])
        yminlab = "{:.1f}".format(min_per_prop[p])
        ymaxlab = "{:.1f}".format(max_per_prop[p])
        ax.set_yticklabels([yminlab, ymaxlab])
        #ax.set_xticklabels([prop_order[i]])
    axs[-1].yaxis.set_ticks([0,1])
    for tick in axs[-1].yaxis.get_major_ticks():
        tick.label20n=True
    yminlab = "{:.1f}".format(min_per_prop[prop_order[-1]])
    ymaxlab = "{:.1f}".format(max_per_prop[prop_order[-1]])
    axs[-1].set_yticklabels([yminlab, ymaxlab])
                         
# Stack subplots together
    plt.subplots_adjust(wspace=0.0)

    plt.show()


#TODO: spring plot also to show results. Color by goodness value (also color
#                                                                 series on
#                                                                 parallel
#                                                                 coordinates by
#                                                                 goodness
#                                                                 values)




