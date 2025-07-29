import numpy as np
import matplotlib.pyplot as plt
import os
import sys

def plot_npz_file(filename1): #plot_npz_file(filename1, filename2):
    data1 = np.load(filename1)
    # data2 = np.load(filename2)
    t = data1['timestamps']
    est = data1['estimated']
    ekf = data1['ekf']
    seen_front = data1['seen_front_f']
    seen_rear = data1['seen_rear_f']
    # gt = data2['gt']

    t = t - t[0]  # start at 0

    plt.figure(figsize=(12, 10))

    for i, label in enumerate(['X', 'Y', 'Z']):
        ax = plt.subplot(4, 1, i + 1)
        ax.plot(t, est[:, i], 'r--', label='Estimated' if i == 0 else "")
        ax.plot(t, ekf[:, i], 'g:', label='EKF' if i == 0 else "")
        #ax.plot(t, gt[:, i], 'b-', label='Ground Truth' if i == 0 else "")
        #ax.ylabel(f'{label} [m]')
	ax.set_ylabel('{} [m]'.format(label))
	
        ax.grid(True)
        if i == 0:
            ax.legend(loc='upper right')

    ax4 = plt.subplot(4, 1, 4)
    ax4.step(t, seen_front, where='post', label='Front Camera', color='orange', linestyle='--')
    ax4.step(t, seen_rear,  where='post', label='Rear Camera',  color='purple', linestyle=':')
    ax4.set_ylabel('Seen')
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(['No', 'Yes'])
    ax4.set_xlabel('Time [s]')
    ax4.grid(True)
    ax4.legend(loc='upper right')
    
    plt.suptitle('Hook Position in Base Link Frame')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # if len(sys.argv) < 2:
        # print("Usage: python plot_hook_base_link.py <path_to_npz_file>")
        # sys.exit(1)

    plot_npz_file(sys.argv[1])
