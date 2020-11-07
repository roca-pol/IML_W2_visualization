# Visualization exercise

## How to run

Run our PCA on a dataset and then reconstruct it:
```bash
python3 main.py pca -d DATASET_NAME
```

Perform a comparison between our PCA, sklearn's PCA and sklearn's IncrementalPCA:
```bash
python3 main.py pca-comparison -d DATASET_NAME
```

Perform a comparison on KMeans performance after running PCA or t-SNE on a dataset:
```bash
python3 main.py kmeans-comparison -d DATASET_NAME
```



\
\
\
Datasets reference:\
DuBois, Christopher L. & Smyth, P. (2008). [UCI Network Data Repository](http://networkdata.ics.uci.edu). Irvine, CA: University of California, School of Information and Computer Sciences.