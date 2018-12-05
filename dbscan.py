import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from matplotlib import image
import cv2
import pandas
import matplotlib.patches as patches


def dbscan(X, spatialeps, coloreps, pts_per_cluster):
    labels = np.ones((X.shape[0], X.shape[1])) * (-1)
    
    c_num = 0
    cluster_count = 0
    Q = []
    neighbor_lists = {}
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if labels[i,j] != -1:
                continue
            cluster, tmp_labels = build_cluster((i,j), X, np.copy(labels), neighbor_lists, c_num, coloreps, spatialeps)
            if len(cluster) >= pts_per_cluster:
                labels = tmp_labels
                c_num += 1
            else:
                neighbor_lists[(i,j)] = cluster
    return labels
            

def build_cluster(start, X, labels, neighbor_lists, C, color_eps, spatial_eps):
    Q = [start]
    cluster = {start}
    while Q:
        pt = Q.pop(0)
        if labels[pt[0], pt[1]] != -1:
            continue
        neighbors = get_neighbors(pt[0], pt[1], X.shape[0], X.shape[1], spatial_eps)
        cur_pt = X[pt[0], pt[1]]
        labels[pt[0], pt[1]] = C
        cluster.add(pt)
        for neighbor in neighbors:
            if np.linalg.norm(cur_pt - X[neighbor[0], neighbor[1]]) < color_eps:
                if neighbor in neighbor_lists:
                    labels[neighbor[0], neighbor[1]] = C
                    cluster.add(neighbor)
                    for n in neighbor_lists[neighbor]:
                        labels[n[0], n[1]] = C
                        cluster.add(n)
                else:
                    Q.append(neighbor)
    return cluster, labels

def get_neighbors(i,j, max_x, max_y, spatial_eps=1):
    i_bdry1 = int(max(i - spatial_eps, 0))
    i_bdry2 = int(min(i+spatial_eps+1, max_x))
    j_bdry1 = int(max(j - spatial_eps, 0))
    j_bdry2 = int(min(j+spatial_eps+1, max_y))
        
    neighbors = set([(x,y) for x in range(i_bdry1, i_bdry2) for y in range(j_bdry1, j_bdry2)])
    neighbors.remove((i,j))
    return list(neighbors)

        
def sklearn_dbscan(lab_img, eps, pts_per_cluster):
    
    indices = np.dstack(np.indices(lab_img.shape[:2]))
    X = np.concatenate((lab_img, indices), axis=-1) 
    X = np.reshape(X, [-1,5])

    db = DBSCAN(eps=eps, min_samples=pts_per_cluster).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(list(labels))) - (1 if -1 in labels else 0)
    print("n_clusters", n_clusters_)
    labels_out = np.ones((lab_img.shape[0], lab_img.shape[1])) * -1
    for i in range(len(labels)):
        labels_out[int(X[i,3]), int(X[i,4])] = labels[i]
    return labels_out
    
    
def get_bounding_boxes(labels, top=10):
    unique, counts = np.unique(labels, return_counts=True)
    bbs = []
    a_sort = np.argsort(counts)
    best = unique[a_sort[-1*top:]]
    for i in best:
        if i == -1:
            continue
        pts = np.where(labels == i)
        bb = []
        bb.append((np.min(pts[0]), np.min(pts[1])))
        bb.append((np.min(pts[0]), np.max(pts[1])))
        bb.append((np.max(pts[0]), np.min(pts[1])))
        bb.append((np.max(pts[0]), np.max(pts[1])))
        bbs.append(bb)
    return bbs

def downsample_to_lab(img, num_times=2):
    n = 0
    labimg = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    while(n<num_times):
        labimg = cv2.pyrDown(labimg)
        n = n+1
    return labimg

def view_dbscan(labels):
    unique, counts = np.unique(labels, return_counts=True)
    n_clusters_ = len(unique) - (1 if -1 in unique else 0)
    n_noise_ = counts[np.where(unique == -1)] if -1 in unique else 0
    # n_clusters_ = len(set(list(labels))) - (1 if -1 in labels else 0)
    # n_noise_ = np.count(labels,list(labels).count(-1)

    print("Number of clusters: ", n_clusters_)

    colors = [[int(plt.cm.Spectral(each)[0]*255),int(plt.cm.Spectral(each)[1]*255), int(plt.cm.Spectral(each)[2]*255)]
              for each in np.linspace(0, 1, n_clusters_)]


    out_img = np.zeros((labels.shape[0], labels.shape[1], 3))
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            if labels[i,j] == -1:
                color = [255, 255, 255]
            else:
                color = colors[int(labels[i,j])]
            out_img[i,j] = color
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(out_img)
    plt.show()

def view_bbs(lab_img, bbs):
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(lab_img, cv2.COLOR_LAB2RGB))
    for bb in bbs:
        rect = patches.Rectangle((bb[0][1],bb[0][0]),abs(bb[2][1]-bb[1][1]),abs(bb[2][0]-bb[1][0]),linewidth=3,edgecolor='r',facecolor='none')
        ax.add_patch(rect)
    plt.show()