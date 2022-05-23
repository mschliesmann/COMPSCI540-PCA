from scipy.linalg import eigh
import numpy as np
import matplotlib.pyplot as plt

def load_and_center_dataset(filename):
    x = np.load(filename)
    xMean = np.mean(x, axis = 0)
    return x - xMean

def get_covariance(dataset):
    return (np.dot(np.transpose(dataset), dataset) / (2413))

def get_eig(S, m):
    x = eigh(S, subset_by_index = [len(S)-m, len(S)-1])
    return np.flip(np.diag(x[0])), np.flip(x[1], axis=1)

def get_eig_prop(S, prop):
    x = eigh(S, eigvals_only=True)
    i = 0
    indexes = []
    while i<1024:
        if x[i]/np.sum(x) > prop:
            indexes.append(i)
        i+=1
    s = eigh(S, subset_by_index = [indexes[0], indexes[len(indexes)-1]])
    return np.flip(np.diag(s[0])), np.flip(s[1], axis=1)

def project_image(image, U):
    columns = np.shape(U)[1]
    i = 0
    sums = 0
    while i < columns:
        projection = np.dot(np.transpose(U[:,i]), image)
        sums += np.dot(projection, U[:,i])
        i+=1
    return sums

def display_image(orig, proj):
    pic1 = np.transpose(np.reshape(orig, (32,32)))
    pic2 = np.transpose(np.reshape(proj, (32,32)))
    fig, axs = plt.subplots(1, 2)
    axs[0].set_title("Original")
    axs[1].set_title("Projection")
    color1 = axs[0].imshow(pic1, aspect = "equal")
    color2 = axs[1].imshow(pic2, aspect = "equal")
    fig.colorbar(color1, ax = axs[0])
    fig.colorbar(color2, ax = axs[1])
    plt.show()
