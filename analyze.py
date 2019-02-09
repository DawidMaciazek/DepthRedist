import pickle
import numpy as np
import matplotlib.pyplot as plt

class extract_m1:
    def __init__(self, pfile, ke_range=[0,100], ke_step=1, z_range=None, z_step=5):
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

        self.z_bin_edges = np.arange(self.z_range[0], self.z_range[1]+10e-6, z_step)
        self.z_bin_centers = self.z_bin_edges - 0.5*z_step
        self.z_bin_centers[0] = np.nan
        self.z_bin_centers = np.append(self.z_bin_centers, np.nan)

        self.ke_ind = np.digitize(self.ke, self.ke_bin_edges, right=True)
        self.z_ind = np.digitize(self.z, self.z_bin_edges, right=True)

    def load(self, pfile):
        with open(pfile, 'rb') as f:
            # [ ke, dx, z ]
            self.data = pickle.load(f)
        self.file_name = pfile.split("/")[-1]
        self.ke = self.data[0]
        self.dx = self.data[1]
        self.z = self.data[2]
        self.sim_cnt = self.data[3]
        print("Total number of simulations: {}".format(self.sim_cnt))

    def calc_vmatrix(self):
        m1_val_ext = np.zeros((len(self.z_bin_centers), len(self.ke_bin_centers)), dtype=float)
        m1_cnt_ext = m1_val_ext.copy()

        for ke_ind, z_ind, dx in zip(self.ke_ind, self.z_ind, self.dx):
            m1_val_ext[z_ind, ke_ind] += dx
            m1_cnt_ext[z_ind, ke_ind] += 1

        # normalize
        m1_val_ext[m1_cnt_ext>0] /= m1_cnt_ext[m1_cnt_ext>0]

        # drop outside range and invert
        m1_val = m1_val_ext[1:-1,1:-1]
        m1_val = m1_val[::-1,:]

        row_labels = self.ke_bin_centers[1:-1]
        inverted_column_labels = self.z_bin_centers[1:-1]

        column_labels = inverted_column_labels[::-1]

        return m1_val, row_labels, column_labels

    def show_hmap(self, saveimg=None):
        m1_val, row_labels, column_labels = self.calc_vmatrix()

        fig, ax1 = plt.subplots(1,1)
        fig.suptitle("File: {} , num os simulations: {}".format(
            self.file_name, self.sim_cnt))
        img = ax1.imshow(m1_val, cmap='hot', vmin=0)

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
        cbar.set_label("Average displacement x-axis [A]")

        if saveimg is not None:
            plt.savefig(saveimg)
        else:
            plt.show()

if __name__ == '__main__':
    # initializing class for analyzing with following bin parameters:
    # range of kinetic energy (0, 20> with 1 step [eV]
    # range of depht (-80, -10> with step 10
    extract = extract_m1("data/100ek_90deg__-5_0.pickle", ke_range=[0,80], ke_step=4, z_step=1)

    # displaying results as heatmap
    a = extract.show_hmap()

    # geting results as matrix for proper analyze
    #m1_matrix, row_labels, column_labels = extract.calc_vmatrix()

