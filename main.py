import sys
import click
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as skPCA
from sklearn.decomposition import IncrementalPCA

import datasets
from analysis import PCA
from cluster import KMeans
from evaluation import print_binary_metrics, print_multi_metrics
import pandas as pd

@click.group()
def cli():
    pass


# -----------------------------------------------------------------------------------------
# Perform PCA on a dataset and then reconstruct it
@cli.command('pca')
@click.option('-d', default='kropt', help='Dataset name kropt | satimage | credita')
def pca(d):
    if d == 'kropt':
        pca_kropt()

    elif d == 'satimage':
        pca_satimage()

    elif d == 'credita':
        pca_credita()

    else:
        raise ValueError('Unknown dataset {}'.format(d))


def pca_kropt():
    X, y = datasets.load_kropt()

    # Divide & Conquer with ScatterMatrix
    sns.set(style="ticks")
    g = sns.pairplot(X)
    g.fig.suptitle('Divide & Conquer - Kropt', y=1)
    g.fig.set_size_inches(10, 10)
    plt.show()

    # Perform custom PCA
    pca = PCA(n_components=2, verbose=True)
    X_trans = pca.fit_transform(X)

    # Reconstruct original dataset
    X_recons = pca.inverse_transform(X_trans)

    # Divide & Conquer for reconstructed data
    sns.set(style="ticks")
    g = sns.pairplot(pd.DataFrame(data=X_recons, columns=X.columns))
    g.fig.suptitle('Divide & Conquer - Reconstructed Kropt', y=1)
    g.fig.set_size_inches(10, 10)
    plt.show()



def pca_satimage():
    X, y = datasets.load_satimage()

    # transform dataset with PCA
    pca = PCA(2, verbose=True)
    X_trans = pca.fit_transform(X)

    # reconstruct original dataset
    X_recons = pca.inverse_transform(X_trans)


    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title('SatImage dataset')
    ax.plot(X.iloc[:, 0].values, X.iloc[:, 1].values, 'o')  # , dataPCA[:,2],'o')
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title('2-components PCA on SatImage')
    ax.plot(X_trans[:, 0], X_trans[:, 1], 'o')  # , dataPCA[:,2],'o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title('Recontructed SatImage dataset')
    ax.plot(X_recons[:, 0], X_recons[:, 1], 'o')  # , dataPCA[:,2],'o')
    ax.set_xlabel('feature 0')
    ax.set_ylabel('feature 1')
    plt.show()

    # fig = plt.figure(figsize=(15, 5))
    # ax = fig.add_subplot(1, 3, 1, projection='3d')
    # ax.set_title('First features')
    # ax.plot3D(X.iloc[:, 0].values, X.iloc[:, 1].values, X.iloc[:, 2].values,'o')  # , dataPCA[:,2],'o')
    # ax = fig.add_subplot(1, 3, 2, projection='3d')
    # ax.set_title('Our PCA')
    # ax.plot3D(X_trans[:, 0], X_trans[:, 1], X_trans[:, 2], 'o')  # , dataPCA[:,2],'o')
    # plt.show()


def pca_credita():
    X, y = datasets.load_credita()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=.11, left=.05, top=.90, right=.95)

    # plot some cols of the original dataset
    col1 = 'A2'
    col2 = 'A3'
    ax[0].scatter(X[col1], X[col2])

    ax[0].title.set_text('Credit-A dataset')
    ax[0].set_xlabel(col1)
    ax[0].set_ylabel(col2)

    # transform dataset with PCA
    pca = PCA(2, verbose=True)
    # pca = skPCA(2)
    X_trans = pca.fit_transform(X)

    ax[1].scatter(X_trans[:, 0], X_trans[:, 1])
    ax[1].title.set_text('Credit-A dataset reduced to 2 dimensions')

    # reconstruct original dataset
    X_recons = pca.inverse_transform(X_trans)

    icol1 = X.columns.get_loc(col1)
    icol2 = X.columns.get_loc(col2)

    ax[2].scatter(X_recons[:, icol1], X_recons[:, icol2])
    
    ax[2].title.set_text('Recontructed Credit-A dataset')
    ax[2].set_xlabel(col1)
    ax[2].set_ylabel(col2)

    plt.show()



# -----------------------------------------------------------------------------------------
# Perform a comparison between our PCA, sklearn's PCA and sklearn's IncrementalPCA
@cli.command('pca-comparison')
@click.option('-d', default='kropt', help='Dataset name kropt | satimage | credita')
def pca_comparison(d):
    if d == 'kropt':
        pca_comparison_kropt()

    elif d == 'satimage':
        pca_comparison_satimage()

    elif d == 'credita':
        pca_comparison_credita()

    else:
        raise ValueError('Unknown dataset {}'.format(d))


def pca_comparison_kropt():
    X, y = datasets.load_kropt()

    # Perform custom PCA, sklearn PCA and IPCA transformations

    pca = PCA(n_components=2, verbose=True)
    X_trans1 = pca.fit_transform(X)

    skpca = skPCA(n_components=2)
    X_trans2 = skpca.fit_transform(X)

    ipca = IncrementalPCA(n_components=2, batch_size=5000)
    X_trans3 = ipca.fit_transform(X)

    # Plot transformed spaces
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    ax[0].scatter(X_trans1[:, 0], X_trans1[:, 1])
    ax[0].title.set_text('Custom PCA Kropt')
    ax[0].set_xlabel('PC1')
    ax[0].set_ylabel('PC2')

    ax[1].scatter(X_trans2[:, 0], X_trans2[:, 1])
    ax[1].title.set_text('Sklearn PCA Kropt')
    ax[1].set_xlabel('PC1')
    ax[1].set_ylabel('PC2')

    ax[2].scatter(X_trans3[:, 0], X_trans3[:, 1])
    ax[2].title.set_text('Sklearn PCA Kropt')
    ax[2].set_xlabel('PC1')
    ax[2].set_ylabel('PC2')

    plt.show()


def pca_comparison_satimage():
    X, y = datasets.load_satimage()

    pca = PCA(2, verbose=True)
    X_trans1 = pca.fit_transform(X)

    skpca = skPCA(2)
    X_trans2 = skpca.fit_transform(X)

    # transform dataset with sklearn's IncrementalPCA
    ipca = IncrementalPCA(2)
    X_trans3 = ipca.fit_transform(X)

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title('SatImage PCA')
    ax.plot(X_trans1[:, 0], X_trans1[:, 1], 'o')  # , dataPCA[:,2],'o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title('SatImage sklearn PCA')
    ax.plot(X_trans2[:, 0], X_trans2[:, 1], 'o')  # , dataPCA[:,2],'o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title('SatImage sklearn IncrementalPCA')
    ax.plot(X_trans3[:, 0], X_trans3[:, 1], 'o')  # , dataPCA[:,2],'o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()

    #plt.show()


def pca_comparison_credita():
    X, y = datasets.load_credita()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=.10, left=.05, top=.90, right=.95)

    # transform dataset with our PCA
    pca = PCA(2, verbose=True)
    X_trans1 = pca.fit_transform(X)

    ax[0].scatter(X_trans1[:, 0], X_trans1[:, 1])
    ax[0].title.set_text('2-component PCA on Credit-A')

    # transform dataset with sklearn's PCA
    skpca = skPCA(2)
    X_trans2 = skpca.fit_transform(X)

    ax[1].scatter(X_trans2[:, 0], X_trans2[:, 1])
    ax[1].title.set_text('2-component PCA (sklearn) on Credit-A')

    # transform dataset with sklearn's IncrementalPCA
    ipca = IncrementalPCA(2)
    X_trans3 = ipca.fit_transform(X)

    ax[2].scatter(X_trans3[:, 0], X_trans3[:, 1])
    ax[2].title.set_text('2-component IncrementalPCA (sklearn) on Credit-A')

    plt.show()



# -----------------------------------------------------------------------------------------
# Perform a comparison on KMeans performance after running PCA or t-SNE on a dataset
@cli.command('kmeans-comparison')
@click.option('-d', default='kropt', help='Dataset name kropt | satimage | credita')
def kmeans_comparison(d):
    if d == 'kropt':
        kmeans_comparison_kropt()

    elif d == 'satimage':
        kmeans_comparison_satimage()

    elif d == 'credita':
        kmeans_comparison_credita()

    else:
        raise ValueError('Unknown dataset {}'.format(d))


def kmeans_comparison_kropt():
    X, y = datasets.load_kropt()
    results = []
    from sklearn.metrics.cluster import davies_bouldin_score
    print(davies_bouldin_score(X, y))

    # Apply custom PCA for dimension reduction
    pca = PCA(n_components=2, verbose=True)
    X_trans = pca.fit_transform(X)

    # Apply K-Means to original data
    kmeans = KMeans(k=18, n_init=50)
    y_pred = kmeans.fit_predict(X)
    results.append(('KMeans', y, y_pred))

    # Apply K-Means to transformed data
    kmeans = KMeans(k=18, n_init=50)
    y_pred_PCA = kmeans.fit_predict(X_trans)
    results.append(('KMeans-PCA', y, y_pred_PCA))

    # t-SNE ------------------------
    # transform dataset with t-SNE
    best_tsne = TSNE(2)
    tsne = best_tsne

    # run several times and keep the best result
    for _ in range(1):  # Too much time duration with big dataset like Kropt
        res = tsne.fit_transform(X)

        if tsne.kl_divergence_ <= best_tsne.kl_divergence_:
            best_tsne = tsne
            X_trans2 = res

        tsne = TSNE(2)

    # run KMeans on reduced dataset
    kmeans = KMeans(k=18, n_init=50)
    y_pred_tsne = kmeans.fit_predict(X_trans2)
    results.append(('KMeans-TSNE', y, y_pred_tsne))

    print('\n\n')
    print_multi_metrics(X, results)

    # Results Visualization

    total_colors = mcolors.cnames
    selected_colors = random.sample(list(total_colors), k=18)

    fig, ax = plt.subplots(2, 3, figsize=(30, 10))

    # PCA result visualization

    cvec = [selected_colors[label] for label in y]

    ax[0, 0].scatter(X_trans[:, 0], X_trans[:, 1], c=cvec, alpha=0.5)
    ax[0, 0].title.set_text('Kropt Groundtruth - Visual PCA')
    ax[0, 0].set_xlabel('PC1')
    ax[0, 0].set_ylabel('PC2')

    cvec = [selected_colors[label] for label in y_pred]

    ax[0, 1].scatter(X_trans[:, 0], X_trans[:, 1], c=cvec, alpha=0.5)
    ax[0, 1].title.set_text('K-Means Clustering - Visual PCA')
    ax[0, 1].set_xlabel('PC1')
    ax[0, 1].set_ylabel('PC2')

    cvec = [selected_colors[label] for label in y_pred_PCA]

    ax[0, 2].scatter(X_trans[:, 0], X_trans[:, 1], c=cvec, alpha=0.5)
    ax[0, 2].title.set_text('K-Means PCA Kropt')
    ax[0, 2].set_xlabel('PC1')
    ax[0, 2].set_ylabel('PC2')

    # t-SNE result visualization

    cvec = [selected_colors[label] for label in y]

    ax[1, 0].scatter(X_trans2[:, 0], X_trans2[:, 1], c=cvec, alpha=0.5)
    ax[1, 0].title.set_text('Kropt Groundtruth - Visual t-SNE')
    ax[1, 0].set_xlabel('PC1')
    ax[1, 0].set_ylabel('PC2')

    cvec = [selected_colors[label] for label in y_pred]

    ax[1, 1].scatter(X_trans2[:, 0], X_trans2[:, 1], c=cvec, alpha=0.5)
    ax[1, 1].title.set_text('K-Means Clustering - Visual t-SNE')
    ax[1, 1].set_xlabel('PC1')
    ax[1, 1].set_ylabel('PC2')

    cvec = [selected_colors[label] for label in y_pred_tsne]

    ax[1, 2].scatter(X_trans2[:, 0], X_trans2[:, 1], c=cvec, alpha=0.5)
    ax[1, 2].title.set_text('K-Means t-SNE Kropt')
    ax[1, 2].set_xlabel('PC1')
    ax[1, 2].set_ylabel('PC2')

    plt.show()


def kmeans_comparison_satimage():

    results = []
    X, y = datasets.load_satimage()
    y = y.values.reshape(-1)

    pca = PCA(2, verbose=True)
    X_trans = pca.fit_transform(X)

    kmeans = KMeans(k=6, n_init=50)
    y_pred = kmeans.fit_predict(X)
    results.append(('KMeans', y, y_pred))

    kmeans = KMeans(k=6, n_init=50)
    y_pred_PCA = kmeans.fit_predict(X_trans)
    results.append(('KMeans-PCA', y, y_pred_PCA))

    # t-SNE ------------------------
    # transform dataset with t-SNE
    best_tsne = TSNE(2)
    tsne = best_tsne
    # run several times and keep the best result
    for _ in range(10):
        res = tsne.fit_transform(X)
        print("B")
        if tsne.kl_divergence_ <= best_tsne.kl_divergence_:
            best_tsne = tsne
            X_trans2 = res

        tsne = TSNE(2)
    # run KMeans on reduced dataset
    kmeans = KMeans(k=6, n_init=50)
    y_pred_tSNE = kmeans.fit_predict(X_trans2)
    results.append(('KMeans-TSNE', y, y_pred_tSNE))

    print(results)
    print('\n\n')
    print_multi_metrics(X, results)

    #PCA figures
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('K-means hist cluster for SatImage')
    ax.hist(y_pred, zorder=3)
    ax.set_xlabel('Number of instances', fontsize=13)
    ax.set_ylabel('Clusters', fontsize=13)
    ax.grid(zorder=0)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('K-means on original SatImage dataset')
    for i in np.unique(y_pred):
        ax.scatter(X_trans[y_pred == i, 0], X_trans[y_pred == i, 1], alpha=.5, color='C' + str(i + 1), label="cluster " + str(i))
    ax.set_xlabel('X', fontsize=13)
    ax.set_ylabel('Y', fontsize=13)
    plt.show()

    #PCA figures
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('K-means hist cluster for SatImage')
    ax.hist(y_pred_PCA, zorder=3)
    ax.set_xlabel('Number of instances', fontsize=13)
    ax.set_ylabel('Clusters', fontsize=13)
    ax.grid(zorder=0)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('K-means on PCA SatImage')
    for i in np.unique(y_pred_PCA):
        ax.scatter(X_trans[y_pred_PCA == i, 0], X_trans[y_pred_PCA == i, 1], alpha=.5, color='C' + str(i + 1), label="cluster " + str(i))
    ax.set_xlabel('X', fontsize=13)
    ax.set_ylabel('Y', fontsize=13)
    plt.show()


    #t-SNE figures
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('K-means t-SNE for SatImage')
    for i in np.unique(y_pred_tSNE):
        ax.scatter(X_trans2[y_pred_tSNE == i, 0], X_trans2[y_pred_tSNE == i, 1], alpha=.5, color='C' + str(i + 1), label="cluster " + str(i))
    ax.set_xlabel('X', fontsize=13)
    ax.set_ylabel('Y', fontsize=13)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('K-means on orignal SatImage dataset')
    for i in np.unique(y_pred):
        ax.scatter(X_trans2[y_pred == i, 0], X_trans2[y_pred == i, 1], alpha=.5, color='C' + str(i + 1), label="cluster " + str(i))
    ax.set_xlabel('X', fontsize=13)
    ax.set_ylabel('Y', fontsize=13)
    plt.show()

    #Ground Turth Figures
    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('SatImage PCA Ground Truth')
    for i in np.unique(y):
        ax.scatter(X_trans[y == i, 0], X_trans[y == i, 1], alpha=.5, color='C' + str(i + 1), label="cluster " + str(i))
    ax.set_xlabel('X', fontsize=13)
    ax.set_ylabel('Y', fontsize=13)
    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('SatImage t-SNE Ground Truth')
    for i in np.unique(y):
        ax.scatter(X_trans2[y == i, 0], X_trans2[y == i, 1], alpha=.5, color='C' + str(i + 1), label="cluster " + str(i))
    ax.set_xlabel('X', fontsize=13)
    ax.set_ylabel('Y', fontsize=13)
    plt.show()


def kmeans_comparison_credita():
    X, y = datasets.load_credita()
    results = []

    # PCA --------------------------
    # transform dataset with our PCA
    pca = PCA(2, verbose=True)
    X_trans1 = pca.fit_transform(X)

    # run KMeans on original dataset
    kmeans = KMeans(k=2, n_init=50)
    y_pred = kmeans.fit_predict(X)
    results.append(('KMeans', y, y_pred))

    # run KMeans on reduced dataset
    kmeans = KMeans(k=2, n_init=50)
    y_pred = kmeans.fit_predict(X_trans1)
    results.append(('KMeans-PCA', y, y_pred))


    # t-SNE ------------------------
    # transform dataset with t-SNE
    best_tsne = TSNE(2)
    tsne = best_tsne

    # run several times and keep the best result
    for _ in range(10):
        res = tsne.fit_transform(X)
        
        if tsne.kl_divergence_ <= best_tsne.kl_divergence_:
            best_tsne = tsne
            X_trans2 = res

        tsne = TSNE(2)

    # run KMeans on reduced dataset
    kmeans = KMeans(k=2, n_init=50)
    y_pred = kmeans.fit_predict(X_trans2)
    results.append(('KMeans-TSNE', y, y_pred))
    
    print('\n\n')
    print_binary_metrics(X, results)


    # plot
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    plt.subplots_adjust(bottom=.10, left=.05, top=.90, right=.95)

    # PCA --------------------------
    ax[0, 0].scatter(X_trans1[y == 0, 0], X_trans1[y == 0, 1])
    ax[0, 0].scatter(X_trans1[y == 1, 0], X_trans1[y == 1, 1])
    ax[0, 0].title.set_text('Credit-A PCA')

    y_pred = results[0][2]
    ax[0, 1].scatter(X_trans1[y_pred == 0, 0], X_trans1[y_pred == 0, 1])
    ax[0, 1].scatter(X_trans1[y_pred == 1, 0], X_trans1[y_pred == 1, 1])
    ax[0, 1].title.set_text('KMeans on original dataset')

    y_pred = 1 - results[1][2]  # we'll invert labels for visualization purposes
    ax[0, 2].scatter(X_trans1[y_pred == 0, 0], X_trans1[y_pred == 0, 1])
    ax[0, 2].scatter(X_trans1[y_pred == 1, 0], X_trans1[y_pred == 1, 1])
    ax[0, 2].title.set_text('KMeans on dataset PCA')


    # t-SNE -------------------------
    ax[1, 0].scatter(X_trans2[y == 0, 0], X_trans2[y == 0, 1])
    ax[1, 0].scatter(X_trans2[y == 1, 0], X_trans2[y == 1, 1])
    ax[1, 0].title.set_text('Credit-A t-SNE')

    y_pred = results[0][2]
    ax[1, 1].scatter(X_trans2[y_pred == 0, 0], X_trans2[y_pred == 0, 1])
    ax[1, 1].scatter(X_trans2[y_pred == 1, 0], X_trans2[y_pred == 1, 1])
    ax[1, 1].title.set_text('KMeans on original dataset')

    y_pred = results[2][2]  # we'll invert labels for visualization purposes
    ax[1, 2].scatter(X_trans2[y_pred == 0, 0], X_trans2[y_pred == 0, 1])
    ax[1, 2].scatter(X_trans2[y_pred == 1, 0], X_trans2[y_pred == 1, 1])
    ax[1, 2].title.set_text('KMeans on dataset t-SNE')

    plt.show()


if __name__ == "__main__":
    cli()

