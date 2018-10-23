import pickle
import numpy as np
import matplotlib.pyplot as plt

class extract_m1:
    def __init__(self, ke_range=[0,100], ke_step=1, z_range=[-60,-20], z_step=5):
        # range=(0,6), step=2  // |-bin edge, *-bin center
        #    |...*...|...*...|...*...|
        #    0   1   2   3   4   5   6
        # 0      1       2       3       4 // bin indexes / discard 0, 4

        self.ke_range = ke_range

        self.ke_bin_edges = np.arange(ke_range[0], ke_range[1]+10e-6, ke_step)
        self.ke_bin_centers = self.ke_bin_edges - 0.5*ke_step
        self.ke_bin_centers[0] = np.nan
        self.ke_bin_centers = np.append(self.ke_bin_centers, np.nan)

        self.z_range = z_range

        self.z_bin_edges = np.arange(z_range[0], z_range[1]+10e-6, z_step)
        self.z_bin_centers = self.z_bin_edges - 0.5*z_step
        self.z_bin_centers[0] = np.nan
        self.z_bin_centers = np.append(self.z_bin_centers, np.nan)

    def load(self, pfile):
        with open(pfile, 'rb') as f:
            # [ ke, dx, z ]
            self.data = pickle.load(f)
        self.ke = self.data[0]
        self.dx = self.data[1]
        self.z = self.data[2]

        self.ke_ind = np.digitize(self.ke, self.ke_bin_edges, right=True)
        self.z_ind = np.digitize(self.z, self.z_bin_edges, right=True)

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

    def show_hmap(self):
        m1_val, row_labels, column_labels = self.calc_vmatrix()

        fix, ax1 = plt.subplots(1,1)
        ax1.imshow(m1_val, cmap='hot', vmin=0)
        plt.show()


if __name__ == '__main__':
    # initializing class for analyzing with following bin parameters:
    # range of kinetic energy (0, 20> with 1 step [eV]
    # range of depht (-80, -10> with step 10
    extract = extract_m1(ke_range=[0,20], ke_step=1, z_range=[-80,-10], z_step=10)

    # loading preprocessed MD data
    extract.load('data/100ek_90deg.pickle')

    # displaying results as heatmap
    extract.show_hmap()

    # geting results as matrix for proper analyze
    m1_matrix, row_labels, column_labels = extract.calc_vmatrix()

