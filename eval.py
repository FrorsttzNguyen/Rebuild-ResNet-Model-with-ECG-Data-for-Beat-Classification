import numpy as np


def evaluate_for_confusion(list_index_gts, list_index_res, num_classes):
    sub_confusion_matrix = np.zeros((4, num_classes))
    sub_confusion_matrix[0, :] = np.sum(np.multiply(list_index_gts, list_index_res), axis=0)
    sub_confusion_matrix[1, :] = np.sum(np.multiply(list_index_gts, np.ones(num_classes) - list_index_res), axis=0)
    sub_confusion_matrix[2, :] = np.sum(np.multiply(list_index_res, np.ones(num_classes) - list_index_gts), axis=0)
    sub_confusion_matrix[3, :] = np.sum(np.multiply(np.ones(num_classes) - list_index_gts, np.ones(num_classes) - list_index_res), axis=0)

    return sub_confusion_matrix
