import sys
import click
import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA as skPCA
from sklearn.decomposition import IncrementalPCA

import datasets
from analysis import PCA
from cluster import KMeans
from evaluation import print_binary_metrics


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


def pca_satimage():
    X, y = datasets.load_satimage()

    # transform dataset with PCA
    pca = PCA(3, verbose=True)
    X_trans = pca.fit_transform(X)

    # reconstruct original dataset
    X_recons = pca.inverse_transform(X_trans)


    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title('SatImage first features')
    ax.plot(X.iloc[:, 0].values, X.iloc[:, 1].values, 'o')  # , dataPCA[:,2],'o')
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title('SatImage PCA')
    ax.plot(X_trans[:, 0], X_trans[:, 1], 'o')  # , dataPCA[:,2],'o')
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title('Recontructed SatImage dataset')
    ax.plot(X_recons[:, 0], X_recons[:, 1], 'o')  # , dataPCA[:,2],'o')
    plt.show()

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1, projection='3d')
    ax.set_title('First features')
    ax.plot3D(X.iloc[:, 0].values, X.iloc[:, 1].values, X.iloc[:, 2].values,'o')  # , dataPCA[:,2],'o')
    ax = fig.add_subplot(1, 3, 2, projection='3d')
    ax.set_title('Our PCA')
    ax.plot3D(X_trans[:, 0], X_trans[:, 1], X_trans[:, 2], 'o')  # , dataPCA[:,2],'o')
    plt.show()


def pca_credita():
    X, y = datasets.load_credita()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=.11, left=.05, top=.90, right=.95)

    # plot some cols of the original dataset
    col1 = 'A2'
    col2 = 'A3'
    ax[0].scatter(X[col1][y == 0], X[col2][y == 0])
    ax[0].scatter(X[col1][y == 1], X[col2][y == 1])

    ax[0].title.set_text(f'Credit-A dataset')
    ax[0].set_xlabel(col1)
    ax[0].set_ylabel(col2)

    # transform dataset with PCA
    pca = PCA(2, verbose=True)
    # pca = skPCA(2)
    X_trans = pca.fit_transform(X)

    ax[1].scatter(X_trans[y == 0, 0], X_trans[y == 0, 1])
    ax[1].scatter(X_trans[y == 1, 0], X_trans[y == 1, 1])
    ax[1].title.set_text(f'Credit-A dataset reduced to 2 dimensions')

    # reconstruct original dataset
    X_recons = pca.inverse_transform(X_trans)

    icol1 = X.columns.get_loc(col1)
    icol2 = X.columns.get_loc(col2)

    ax[2].scatter(X_recons[y == 0, icol1], X_recons[y == 0, icol2])
    ax[2].scatter(X_recons[y == 1, icol1], X_recons[y == 1, icol2])
    
    ax[2].title.set_text(f'Recontructed Credit-A dataset')
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


def pca_comparison_satimage():
    X, y = datasets.load_satimage()

    pca = PCA(3, verbose=True)
    X_trans1 = pca.fit_transform(X)

    skpca = skPCA(3)
    X_trans2 = skpca.fit_transform(X)

    # transform dataset with sklearn's IncrementalPCA
    ipca = IncrementalPCA(3)
    X_trans3 = ipca.fit_transform(X)

    fig = plt.figure(figsize=(15, 5))
    ax = fig.add_subplot(1, 3, 1)
    ax.set_title('SatImage PCA')
    ax.plot(X_trans1[:, 0], X_trans1[:, 1], 'o')  # , dataPCA[:,2],'o')
    ax = fig.add_subplot(1, 3, 2)
    ax.set_title('SatImage sklearn PCA')
    ax.plot(X_trans2[:, 0], X_trans2[:, 1], 'o')  # , dataPCA[:,2],'o')
    ax = fig.add_subplot(1, 3, 3)
    ax.set_title('SatImage sklearn IncrementalPCA')
    ax.plot(X_trans3[:, 0], X_trans3[:, 1], 'o')  # , dataPCA[:,2],'o')
    plt.show()

    plt.show()


def pca_comparison_credita():
    X, y = datasets.load_credita()

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    plt.subplots_adjust(bottom=.10, left=.05, top=.90, right=.95)

    # transform dataset with our PCA
    pca = PCA(2, verbose=True)
    X_trans1 = pca.fit_transform(X)

    ax[0].scatter(X_trans1[y == 0, 0], X_trans1[y == 0, 1])
    ax[0].scatter(X_trans1[y == 1, 0], X_trans1[y == 1, 1])
    ax[0].title.set_text(f'Credit-A PCA')

    # transform dataset with sklearn's PCA
    skpca = skPCA(2)
    X_trans2 = skpca.fit_transform(X)

    ax[1].scatter(X_trans2[y == 0, 0], X_trans2[y == 0, 1])
    ax[1].scatter(X_trans2[y == 1, 0], X_trans2[y == 1, 1])
    ax[1].title.set_text(f'Credit-A sklearn PCA')

    # transform dataset with sklearn's IncrementalPCA
    ipca = IncrementalPCA(2)
    X_trans3 = ipca.fit_transform(X)

    ax[2].scatter(X_trans3[y == 0, 0], X_trans3[y == 0, 1])
    ax[2].scatter(X_trans3[y == 1, 0], X_trans3[y == 1, 1])
    ax[2].title.set_text(f'Credit-A sklearn IncrementalPCA')

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


def kmeans_comparison_satimage():
    X, y = datasets.load_satimage()


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
    tsne = TSNE(2)
    X_trans2 = tsne.fit_transform(X)

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
    ax[0, 0].title.set_text(f'Credit-A PCA')

    y_pred = results[0][2]
    ax[0, 1].scatter(X_trans1[y_pred == 0, 0], X_trans1[y_pred == 0, 1])
    ax[0, 1].scatter(X_trans1[y_pred == 1, 0], X_trans1[y_pred == 1, 1])
    ax[0, 1].title.set_text(f'KMeans on original dataset')

    y_pred = 1 - results[1][2]  # we'll invert labels for visualization purposes
    ax[0, 2].scatter(X_trans1[y_pred == 0, 0], X_trans1[y_pred == 0, 1])
    ax[0, 2].scatter(X_trans1[y_pred == 1, 0], X_trans1[y_pred == 1, 1])
    ax[0, 2].title.set_text(f'KMeans on dataset PCA')


    # t-SNE -------------------------
    ax[1, 0].scatter(X_trans2[y == 0, 0], X_trans2[y == 0, 1])
    ax[1, 0].scatter(X_trans2[y == 1, 0], X_trans2[y == 1, 1])
    ax[1, 0].title.set_text(f'Credit-A t-SNE')

    y_pred = results[0][2]
    ax[1, 1].scatter(X_trans2[y_pred == 0, 0], X_trans2[y_pred == 0, 1])
    ax[1, 1].scatter(X_trans2[y_pred == 1, 0], X_trans2[y_pred == 1, 1])
    ax[1, 1].title.set_text(f'KMeans on original dataset')

    y_pred = 1 - results[2][2]  # we'll invert labels for visualization purposes
    ax[1, 2].scatter(X_trans2[y_pred == 0, 0], X_trans2[y_pred == 0, 1])
    ax[1, 2].scatter(X_trans2[y_pred == 1, 0], X_trans2[y_pred == 1, 1])
    ax[1, 2].title.set_text(f'KMeans on dataset t-SNE')

    plt.show()


if __name__ == "__main__":
    cli()

