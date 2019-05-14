import sys
import glob
import pickle
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['errorbar.capsize'] = 3


class extract_m1:
    def __init__(self, pfile, ke_range=[0,100], ke_step=1, z_range=None,
                 z_step=5, calc_mom=False, vmax=None):
        # range=(0,6), step=2  // |-bin edge, *-bin center
        #    |...*...|...*...|...*...|
        #    0   1   2   3   4   5   6
        # 0      1       2       3       4 // bin indexes / discard 0, 4

        self.load(pfile)

        self.ke_range = ke_range

        self.ke_bin_edges = np.arange(ke_range[0], ke_range[1]+10e-6, ke_step)
        self.ke_bin_centers = self.ke_bin_edges - 0.5*ke_step
        self.ke_bin_centers[0] = np.nan
        self.ke_bin_centers = np.append(self.ke_bin_centers, np.nan)

        if z_range is not None:
            self.z_range = z_range
        else:
            center = int(np.round(np.mean(self.z)))
            self.z_range = [center-15, center+15]

        self.z_bin_edges = np.arange(self.z_range[0], self.z_range[1]+10e-6,
                                     z_step)
        self.z_bin_centers = self.z_bin_edges - 0.5*z_step
        self.z_bin_centers[0] = np.nan
        self.z_bin_centers = np.append(self.z_bin_centers, np.nan)

        self.ke_ind = np.digitize(self.ke, self.ke_bin_edges, right=True)
        self.z_ind = np.digitize(self.z, self.z_bin_edges, right=True)

        self.calc_mom = calc_mom
        self.vmax = vmax

    def load(self, pfile):
        """Load the data from file pfile."""
        with open(pfile, 'rb') as f:
            # [ ke, dx, z ]
            self.data = pickle.load(f)
        self.file_name = pfile.split("/")[-1]
        self.ke = self.data[0]
        self.dx = self.data[1]
        self.z = self.data[2]
        self.recoil_index = self.data[3]
        try:
            self.sim_cnt = self.data[4]
            print("n={}/{} ({})".format(self.sim_cnt,
                                        len(set(self.recoil_index)), pfile))
        except IndexError:
            self.recoil_index = None
            self.sim_cnt = self.data[3]
            print("n={} ({})".format(self.sim_cnt, pfile))
        print("dx_min={:.2f}, dx_max={:.2f}".format(self.dx.min(),
                                                    self.dx.max()))
        
    def calc_vmatrix(self):
        """Calculate a 2d histogram over recoil depth and energy."""
        m1_val_ext = np.zeros((len(self.z_bin_centers),
                               len(self.ke_bin_centers)), dtype=float)
        m1_cnt_ext = m1_val_ext.copy()

        for ke_ind, z_ind, dx in zip(self.ke_ind, self.z_ind, self.dx):
            m1_val_ext[z_ind, ke_ind] += dx
            m1_cnt_ext[z_ind, ke_ind] += 1

        # normalize
        if self.calc_mom:
           m1_val_ext[m1_cnt_ext>0] /= self.sim_cnt
        else:
           m1_val_ext[m1_cnt_ext>0] /= m1_cnt_ext[m1_cnt_ext>0]

        # drop outside range and invert
        m1_val = m1_val_ext[1:-1,1:-1]
        m1_val = m1_val[::-1,:]

        row_labels = self.ke_bin_centers[1:-1]
        inverted_column_labels = self.z_bin_centers[1:-1]

        column_labels = inverted_column_labels[::-1]

        return m1_val, row_labels, column_labels

    def show_hmap(self, z1=None, z2=None, saveimg=None):
        """Show the 2d histogram as a heat map."""
        m1_val, row_labels, column_labels = self.calc_vmatrix()
        print column_labels

        fig, ax1 = plt.subplots(1,1)
        fig.suptitle("File: {} , num of simulations: {}".format(
            self.file_name, self.sim_cnt))
        img = ax1.imshow(m1_val, cmap='hot', vmin=0, vmax=self.vmax)

        row_map_dict = dict(zip(range(len(row_labels)), row_labels))
        def row_mapper(val, n):
            val = int(val)
            if val in row_map_dict:
                return row_map_dict[val]
            return 'na'
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(row_mapper))
        ax1.set_xlabel("Recoil peak kinetic energy [eV]")

        col_map_dict = dict(zip(range(len(column_labels)), column_labels))
        def column_mapper(val, n):
            val = int(val)
            if val in col_map_dict:
                return col_map_dict[val]
            return 'na'

        ax1.yaxis.set_major_formatter(plt.FuncFormatter(column_mapper))
        ax1.set_ylabel("Recoil initial z position [A]")

        cbar = plt.colorbar(img)
        if self.calc_mom:
            cbar.set_label("Contribution to first moment [A]")
        else:
            cbar.set_label("Average displacement x-axis [A]")

        f = interpolate.interp1d(column_labels, np.arange(len(column_labels)))
        if column_labels[0] > 0 and column_labels[-1] < 0:
            ax1.axhline(y=f(0), color='white', linewidth=1)
        if z1 is not None:
            ax1.axhline(y=f(z1), color='white', linewidth=1, linestyle='--')
        if z2 is not None:
            ax1.axhline(y=f(z2), color='white', linewidth=1, linestyle='--')

        if saveimg is not None:
            plt.savefig(saveimg)
        else:
            plt.show()

    def calc_m1_distrib(self):
        """Calculate a histogram of m1 values per primary recoil,
        its mean value, and the standard deviation of the mean value."""
        m1_dict = {}
        for dx, recoil_index in zip(self.dx, self.recoil_index):
            if recoil_index in m1_dict:
                m1_dict[recoil_index] += dx
            else:
                m1_dict[recoil_index] = dx
                
        m1 = np.mean(m1_dict.values())
        m1_err = np.std(m1_dict.values()) / np.sqrt(len(m1_dict))
        
        return m1_dict, m1, m1_err

    def calc_1d_distrib(self):
        """Calculate the 1d histogram over energy."""
        m1_val, row_labels, column_labels = self.calc_vmatrix()
        m1_val = np.sum(m1_val, axis=0)
        
        return m1_val, row_labels    

    def show_1d_distrib(self, compareto=None, saveimg=None):
        """Show the 1d histogram and optionally compare to second histogram."""
        if not self.calc_mom:
            sys.exit("Cannot construct histogram of average displacement.")
        m1_val, labels = self.calc_1d_distrib()
        m1_sum = np.sum(m1_val)
        print 'm1=', m1_sum

        fig, ax1 = plt.subplots(1,1)
        label = "{} (M1={:.1f})".format(self.file_name, m1_sum)
        ax1.plot(m1_val, label=label)

        row_map_dict = dict(zip(range(len(labels)), labels))
        def row_mapper(val, n):
            val = int(val)
            if val in row_map_dict:
                return row_map_dict[val]
            return 'na'
        ax1.xaxis.set_major_formatter(plt.FuncFormatter(row_mapper))
        ax1.set_xlabel("Recoil peak kinetic energy [eV]")

        ax1.set_ylim(0, self.vmax)
        ax1.set_ylabel("Contribution to first moment [A]")

        # compare to other data
        if compareto is not None:
            m1_val, labels = compareto.calc_1d_distrib()
            m1_sum = np.sum(m1_val)
            label = "{} (M1={:.1f})".format(compareto.file_name, m1_sum)
            ax1.plot(m1_val, '--', label=label)
            print 'm1=', m1_sum

        ax1.legend(loc='best')

        if saveimg is not None:
            plt.savefig(saveimg)
        else:
            plt.show()

def show_hmap(fname, z1, z2):
    """Show the 2d histogram over recoil depth and energy as a heat map
    for one data set."""
    extract = extract_m1(fname, 
                         ke_range=[0,96], ke_step=4, z_step=1, 
                         calc_mom=True, vmax=None)
    extract.show_hmap(z1, z2)


def show_1d_distrib(fname1, fname2=None):
    """Show the 1d histogram over recoil energy for one or two data sets."""
    extract = extract_m1(fname1, 
                         ke_range=[0,120], ke_step=4, z_step=1, 
                         calc_mom=True, vmax=None)
    if fname2 is None:
        extract2 = None
    else:
        extract2 = extract_m1(fname2, 
                              ke_range=[0,120], ke_step=4, z_step=1, 
                              calc_mom=True, vmax=None)
    extract.show_1d_distrib(extract2)


def show_m1_vs_z(*fpatterns):
    """Show m1 as a function of depth using files adhering to fpattern.
    fpattern must be of the form prefix*_*suffix, where the '*' represent
    the depth limits."""
    for fpattern in fpatterns:
        fnames = glob.glob(fpattern)
        prefix, suffix = fpattern.split('*_*')
        z = []
        m1 = []
        m1_err = []
        for fname in fnames:
            zz = fname.replace(prefix, '', 1).replace(suffix, '')
            try:
                z1, z2 = zz.split('_')
                z1 = float(z1)
                z2 = float(z2)
            except ValueError:
                continue
            z.append(0.5 * (z1 + z2))
            extract = extract_m1(fname,
                                 ke_range=[0,120], ke_step=4, z_step=1, 
                                 calc_mom=True, vmax=None)
            if extract.recoil_index is None: # we cannot use calc_m1_distrib
                m1_val, _ = extract.calc_1d_distrib()
                m1.append(np.sum(m1_val))
            else: # we could also use calc_1d_distrib
                _, m1_val, m1_val_err = extract.calc_m1_distrib()
                m1.append(m1_val)
                m1_err.append(m1_val_err)
        
        if len(z) > 3:
            linestyle='.-'
        else:
            linestyle='.'
        if len(m1) == len(m1_err):
            z, m1, m1_err = zip(*sorted(zip(z, m1, m1_err)))
            plt.errorbar(-np.array(z), m1, yerr=m1_err,
                         fmt=linestyle, label=fpattern)
        else:
            z, m1 = zip(*sorted(zip(z, m1)))
            plt.plot(-np.array(z), m1, linestyle, label=fpattern)
    plt.ylim(0, extract.vmax)
    plt.xlabel('Depth (A)')
    plt.ylabel(r'M$_1$ (A)')
    plt.legend(loc='best')
    plt.show()


def show_m1_hist(fname):
    """Show a histogram of m1 values caused by primary recoils."""
    extract = extract_m1(fname,
                         ke_range=[0,120], ke_step=4, z_step=1, 
                         calc_mom=True, vmax=None)
    if extract.recoil_index is None: # we cannot use calc_m1_distrib
        sys.exit('Cannot calculate m1 histogram.')
    m1_dict, m1_mean, m1_mean_err = extract.calc_m1_distrib()
    plt.hist(m1_dict.values(), 
             label='mean={:.2f}(+/-{:.2f})'.format(m1_mean, m1_mean_err))
    plt.xlabel(r'M$_1$ (single recoil cascade)')
    plt.ylabel('count')
    plt.legend()
    plt.title(fname)
    plt.show()

if __name__ == '__main__':

#    show_1d_distrib("data_comparison_time/100ek__-15_-10_UP_4500.pickle",
#                    "data_comparison_time/100ek__-15_-10_UP_3000.pickle")

#    show_hmap("data_comparison_time/100ek__-15_-10_DOWN_4500.pickle", -15, -10)

#    show_m1_vs_z('data/100ek_90deg__*_*.pickle',
#                 'data_30cut/100ek_90deg_cut__*_*.pickle',
#                 'data_comparison_time/100ek__*_*_NORMAL_4500.pickle',
#                 'data_comparison_time/100ek__*_*_NORMAL_3000.pickle',
#                 'data_relax/100ek_relax__*_*.pickle',
#                 'data_small/100ek_small__*_*.pickle',
#                 'data_small_NOrelax/100ek_small_NOrelax__*_*.pickle')

    show_m1_hist('data_comparison_time/100ek__-15_-10_UP_4500.pickle')
