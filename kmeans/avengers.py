import numpy as np
import cv2
import matplotlib.gridspec as gridspec

from matplotlib import pyplot as plt
from numpy.matlib import repmat
from cluster import KMeans


def blend(img, labels, img_background, fx=0.60, fy=0.60):

    img = cv2.resize(img, None, fx=fx, fy=fy)

    labels = labels.astype(np.float32) * 255
    labels = np.repeat(np.expand_dims(labels, 2), repeats=3, axis=2)
    labels = cv2.resize(labels, None, fx=fx, fy=fy)
    labels = labels[:, :, 0:1]
    labels = (labels / 255.).astype(np.uint8)

    h1, w1, _ = img.shape
    h2, w2, _ = img_background.shape

    img_to_blend = np.pad(img, pad_width=(
        (int(h2 - h1), 0),
        (int((w2 - w1) / 2) + (w2-w1)%2, int((w2 - w1) / 2)), (0, 0)),
           mode='constant')

    labels_to_blend = np.pad(labels, pad_width=(
        (int(h2 - h1), 0),
        (int((w2 - w1) / 2) + (w2-w1)%2, int((w2 - w1) / 2)), (0,0)),
                          mode='constant')

    img_background = cv2.blur(img_background, (3, 3))
    blended = labels_to_blend * img_to_blend + (1. - labels_to_blend) * img_background
    blended = blended.astype(np.uint8)

    return blended


def main_kmeans_img(img_path, background_path):
    """
    Main function to run kmeans for image segmentation.

    Parameters
    ----------
    img_path: str
        Path of the image to load and segment.

    Returns
    -------
    None
    """

    # load the image
    img = np.float32(cv2.imread(img_path))
    print(img.shape)
    h, w, c = img.shape

    fig = plt.figure(constrained_layout=True, figsize=(14, 6))
    spec = gridspec.GridSpec(ncols=4, nrows=2, figure=fig)

    ax1 = fig.add_subplot(spec[0, 0])
    ax1.set_title('Input image')

    ax2 = fig.add_subplot(spec[1, 0])
    ax2.set_title('M - K-Means Segmentation Mask')

    ax3 = fig.add_subplot(spec[0:, 1:])
    ax3.set_title('Avengers * M + DIEF * (1-M)')

    ax1.axis('off')
    ax2.axis('off')
    ax3.axis('off')

    ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8))
    plt.waitforbuttonpress()

    # add coordinates
    row_indexes = np.arange(0, h)
    col_indexes = np.arange(0, w)
    coordinates = np.zeros(shape=(h, w, 2))
    coordinates[..., 0] = repmat(row_indexes, w, 1).T / w
    coordinates[..., 1] = repmat(col_indexes, h, 1) / h
    img = img / 255

    # data = np.concatenate((img, coordinates), axis=-1)
    data = np.reshape(img, newshape=(w * h, 3))

    # solve kmeans optimization
    initial_centers = np.array([[0, 1, 0], [0.5, 0.5, 0.5]])
    labels = KMeans(n_cl=2, verbose=False,
                    initial_centers=initial_centers).fit_predict(data)
    labels = np.reshape(labels, (h, w))

    ax2.imshow(labels, cmap='hot')
    plt.waitforbuttonpress()

    img_background = np.float32(cv2.imread(background_path))
    h, w, c = img.shape

    data = np.reshape(data, (h, w, 3)) * 255

    result = blend(data, labels, img_background)
    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB).astype(np.uint8)

    ax3.imshow(result)
    plt.waitforbuttonpress()