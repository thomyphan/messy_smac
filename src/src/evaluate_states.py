from smac.env import StarCraft2Env
import numpy as np
import sys
from os import listdir
from os.path import isfile, join
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, SparsePCA, TruncatedSVD, KernelPCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plot

map_name = sys.argv[1]
use_tsne = False
plot_observations = True
noisy = False
if len(sys.argv) > 2:
    noisy = True

if noisy:
    filename_prefix = "nobservations_{}_".format(map_name)
else:
    filename_prefix = "observations_{}_".format(map_name)

dirname = "../init_states"

files = [join(dirname, f) for f in listdir(dirname) if isfile(join(dirname, f)) and f.startswith(filename_prefix)]

episodes = []
available_labels = list(range(10))
x = []
y = []
for file in files:
    new_episodes = list(np.load(file, allow_pickle=True))
    episodes += new_episodes
    for episode in new_episodes:
        for label in available_labels:
            if label < len(episode):
                x.append(np.array(episode[label]).reshape(-1))
                y.append(label)

episode_lengths = [len(episode) for episode in episodes]
print("Avg. episode length:", np.mean(episode_lengths))
print("Min. episode length:", np.min(episode_lengths))
print("Max. episode length:", np.max(episode_lengths))

x = np.array(x)
x = StandardScaler().fit_transform(x)
y = np.array(y)

if use_tsne:
    tsne = TSNE(n_components=2)
    principal_components = tsne.fit_transform(x)
else:
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(x)

pc_1 = [e[0] for e in principal_components]
pc_2 = [e[1] for e in principal_components]

plot.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plot.tick_params(
    axis='y',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    right=False,
    left=False,
    labelleft=False) # labels along the bottom edge are off
plot.scatter(pc_1, pc_2, c=y, cmap='plasma', alpha=0.5)
cbar = plot.colorbar()
cbar.ax.get_yaxis().labelpad = 15
cbar.ax.set_ylabel('number of random time steps', rotation=270)
plot.title(map_name)
plot.show()