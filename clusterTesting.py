# Author: Nikoleta Chantzi
# Paper: https://papers.nips.cc/paper/2021/file/22785dd2577be2ce28ef79febe80db10-Paper.pdf
# Project: Brain Region Correlation in the Context of Smart Homes
# CASAS DataSet: https://casas.wsu.edu/datasets/
# For this particular implementation: HH101


from sklearn.cluster import KMeans, Birch, BisectingKMeans
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer
from sklearn.decomposition import PCA

def kMeansTesting(k):
    # find smaller dataset
    kmeans = KMeans(n_clusters=k, random_state=0, n_init="auto").fit(dataset.iloc[:,:-1])
    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    dataset.loc[:, "label"] =labels
    # print(dataset[["activity", "label"]].drop_duplicates())
    labelsCount = len(np.unique(labels))
    # print(labelsCount)

    ## heatmap
    piv = pd.pivot_table(dataset, values="label", index=["lastSensorEventHours"], columns=["activity"], fill_value=0)
    ax = sns.heatmap(piv, square=True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.tight_layout()
    plt.legend(title="Clusters")
    plt.title('K-means Heatmap- Matching Labels with Clusters')
    plt.savefig('kMeans_actLabels_heatmap.png')
    plt.show()

    ## scatter
    fig, ax = plt.subplots()
    plt.scatter(dataset["activity"], dataset["label"])
    ax.set_xticks(ax.get_xticks()[::2])
    plt.ylabel('Clusters')
    plt.title('K-means - Matching Labels with Clusters')
    plt.savefig('kMeans_actLabels.png')
    plt.show()
    return labels, centroids

def bisectingKMeansTesting(k):
    bisect_means = BisectingKMeans(n_clusters=k, random_state=0).fit(dataset.iloc[:,:-1])
    labels = bisect_means.labels_
    dataset.loc[:, "label"] =labels
    fig, ax = plt.subplots()
    plt.scatter(dataset["activity"], dataset["label"])
    plt.xticks(rotation='vertical')
    ax.set_xticks(ax.get_xticks()[::2])
    plt.ylabel('Clusters')
    plt.title('Bisecting K-means - Matching Labels with Clusters')
    plt.savefig('bisectingkMeans_actLabels.png')
    plt.show()

    ## heatmap
    piv = pd.pivot_table(dataset, values="label", index=["lastSensorEventHours"], columns=["activity"], fill_value=0)
    ax = sns.heatmap(piv, square=True)
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=90)
    plt.tight_layout()
    plt.title('Bisecting K-means Heatmap - Matching Labels with Clusters')
    plt.savefig('bisectingkMeans_actLabels_heatmap.png')
    plt.show()
    return bisect_means.labels_

def elbowMethod():
    model = KMeans()
    # k is range of number of clusters.
    visualizer = KElbowVisualizer(model, k=(2, 40), timings=True)
    cluster_df = dataset.iloc[:, :-1]
    # print(cluster_df)
    visualizer.fit(cluster_df)  # Fit data to visualizer
    plt.savefig('elbowMethod.png')
    visualizer.show()  # Finalize and render figure


# Gap Statistic for K means; takes forever to run!
def optimalK(data, nrefs, maxClusters):
    """
    Calculates KMeans optimal K using Gap Statistic
    Params:
        data: ndarry of shape (n_samples, n_features)
        nrefs: number of sample reference datasets to create
        maxClusters: Maximum number of clusters to test for
    Returns: (gaps, optimalK)
    """
    gaps = np.zeros((len(range(1, maxClusters)),))
    resultsdf = pd.DataFrame({'clusterCount': [], 'gap': []})
    for gap_index, k in enumerate(range(1, maxClusters)):
        # Holder for reference dispersion results
        refDisps = np.zeros(nrefs)
        # For n references, generate random sample and perform kmeans getting resulting dispersion of each loop
        for i in range(nrefs):
            # Create new random reference set
            randomReference = np.random.random_sample(size=data.shape)

            # Fit to it
            km = KMeans(k)
            km.fit(randomReference)

            refDisp = km.inertia_
            refDisps[i] = refDisp
        # Fit cluster to original data and create dispersion
        km = KMeans(k)
        km.fit(data)

        origDisp = km.inertia_
        # Calculate gap statistic
        gap = np.log(np.mean(refDisps)) - np.log(origDisp)
        # Assign this loop's gap statistic to gaps
        gaps[gap_index] = gap

        resultsdf = resultsdf.append({'clusterCount': k, 'gap': gap}, ignore_index=True)


    return (gaps.argmax() + 1, resultsdf)

# reduce to two dimensions so as to visualize results
def visualizePCA():
    ### Run PCA on the data and reduce the dimensions in pca_num_components dimensions
    pca_num_components = 2
    dataset_to_reduce = dataset.drop(['activity', 'label'], axis=1)
    # print(dataset_to_reduce)
    reduced_data = PCA(n_components=pca_num_components).fit_transform(dataset_to_reduce)
    results = pd.DataFrame(reduced_data, columns=['pca1', 'pca2'])
    sns.scatterplot(x="pca1", y="pca2", hue= labels, data=results)
    plt.legend(title="Clusters")
    plt.title('K-means Clustering with 2 dimensions')
    plt.savefig('pcaVisualize.png')
    plt.show()

if __name__ == "__main__":
    # dataset = np.loadtxt("/Users/nikoletachantzi/Library/CloudStorage/OneDrive-DartmouthCollege/WINTER 2023/SPLICE/hh101/hh101.ann.features.csv",names=False)
    # print(dataset)
    dataset = pd.read_csv("hh101/hh101.ann.features.csv", delimiter=',')

    # check unique labels to get k_init; returns 35
    # uniqueLabelsCount = dataset.iloc[:, -1].unique().size
    # print(uniqueLabelsCount)

    # check for nan values; returns 0
    # nan_count = dataset.isna().sum()
    # print(nan_count)

    # filter through NaNs; depending on which feature we're talking about; returns 0
    # nan_count = dataset.isna().sum().sum()
    # print(nan_count)

    # obtain optimal k; returns 10; takes a while to run
    # elbowMethod()
    k = 10

    # test it on kMeans or, alternatively bisectingKMeans; uncomment once at a time
    labels, centroids = kMeansTesting(k)
    # labels = bisectingKMeansTesting(k)
    visualizePCA()
