from sklearn.datasets import load_digits
from sklearn.manifold._t_sne import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import argparse
import os

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--features_file', type=str, default='fatures_dir/', help='file with the features, last column must be the class')
    parser.add_argument('--img_output', type=str, default='example.pdf', help='image output')


    opt = parser.parse_args()
    return opt

opt = parse_opt()

features_y =  np.load(os.path.join(opt.features_file))

y = features_y[:,-1]
features = features_y[:,:-1]

classes=['P. platessa', 'S. solea', 'E. gurnardus', 'L. limanda', 'A. radiata', 'M. merlangus', 'S. canicula']
labels = []

for l in y:
    specie = ''
    if l == 999:
        specie = 'Unknown'
    else:
        specie = classes[int(l)]

    labels.append(specie)

print(np.unique(y))
#classes = np.unique(y)
print(np.unique(labels))
n_classes = len(np.unique(labels)) #+1 because of the unknown
sns.set(rc={'figure.figsize':(11.7,8.27)})
palette = sns.color_palette("bright", n_classes)


tsne = TSNE()
X_embedded = tsne.fit_transform(features)


sns.scatterplot(x=X_embedded[:,0], y=X_embedded[:,1], hue=labels, legend='full', palette=palette, hue_order=sorted(np.unique(labels)))
plt.savefig(opt.img_output, format="pdf", bbox_inches="tight")


