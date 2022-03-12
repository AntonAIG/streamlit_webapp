# this is a basic machinel learning web deployment using streamlit.
# It was completed using SVM, SVC, and Random Forest classifiers using
# datasets available in sklearn

import streamlit as st
from sklearn import datasets
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA


st.title("Streamlit example")

st.write("""
# Explore different classifier
Which one is the best?
""")


dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", "Brest Cancer", "Wine Dataset"))
st.write(dataset_name)

# create a selection box for the different classifiers
clf_name = st.sidebar.selectbox("Select Clasifier", ("KNN", "SVM", "Random Forest"))
st.write(clf_name)


# create a function that gets the datasets
def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Brest Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    X = data.data
    y = data.target
    return X, y


X, y = get_dataset(dataset_name)

st.write("Shape of dataset", X.shape)
st.write("Number of classes", len(np.unique(y)))


def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params["K"] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 100.0)
        params["C"] = C
    else:
        max_depth = st.sidebar.slider("Max Depth", 2, 10)
        n_estimators = st.sidebar.slider("Number of Estimators", 1, 100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
    return params


params = add_parameter_ui(clf_name)


def get_classifier(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors=params["K"])
    elif clf_name == "SVM":
        clf = SVC(C=params["C"])
    else:
        clf = RandomForestClassifier(max_depth=params["max_depth"],
                                     n_estimators=params["n_estimators"],
                                     random_state=0)
    return clf


clf = get_classifier(clf_name, params)


# Classification solving
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

acc = accuracy_score(y_test, y_pred)
st.write(f"Classifier = {clf_name}")
st.write(f"Accuracy = {acc}")


# ploting the output
# using PCA as a feature reduction method that projects our features to a lower dimension
pca = PCA(2)
X_projected = pca.fit_transform(X)
x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap="PiYG")
plt.xlabel("Principal component 1")
plt.ylabel("Principal component 2")
plt.colorbar()

st.pyplot(fig)