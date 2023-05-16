import numpy as np
import time
import os

def max_clique(W):
    flat_matrix = np.array(W).flatten()
    mean_flat_matrix = np.mean(flat_matrix)
    v = np.sum(np.multiply(W, (W > mean_flat_matrix)), axis=1)
    sorted_v = np.argsort(v)[::-1]
    return sorted_v[:4]


def get_best_gain(n, vertex_list, triangle, W, no_vertex_list):
    gvec = None

    if no_vertex_list is None:
        original_range = range(n)
        no_vertex_list = np.setdiff1d(original_range, vertex_list)

    for tr in triangle:
        column = W[:][tr]
        column[no_vertex_list] = 0

        if gvec is None:
            gvec = column
        else:
            gvec = gvec + column

    index_max = np.argmax(gvec)
    max_element = np.max(gvec)
    return index_max, max_element


def tic():
    global startTime_for_tictoc
    startTime_for_tictoc = time.time()


def toc():
    if 'startTime_for_tictoc' in globals():
        print("Elapsed time is " + str(time.time() - startTime_for_tictoc) + " seconds.")
    else:
        print("Toc: start time not set")


def delete_empty_subdirectories(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        for dirname in dirnames:
            dir_full_path = os.path.join(dirpath, dirname)
            try:
                os.rmdir(dir_full_path)
                print(f"Deleted empty directory: {dir_full_path}")
            except OSError:
                pass


def read_file(file_path):
    tuples = []
    with open(file_path, 'r') as file:
        for line in file:
            args = line.strip().split()
            if len(args) == 4:
                # Create a tuple and append it to the list
                tuples.append(tuple(args))
            else:
                print(f"Invalid line format: {line}")
    return tuples