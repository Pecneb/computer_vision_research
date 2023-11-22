import timeit
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm
from scipy.spatial.distance import cdist


def euclidean_distance(a, b):
    return (((a[0]-b[0])**2)+((a[1]-b[1])**2))**0.5


def euclidean_distance_np_linalg(a, b):
    return np.linalg.norm(a-b)


def euclidean_distance_np_sqrt(a, b):
    return np.sqrt(np.sum((a-b)**2))


def euclidean_distance_np_sqrt_square(a, b):
    return np.sqrt(np.sum(np.square(a-b)))


def euclidean_distance_np_dot(a, b):
    return np.sqrt((a-b).T @ (a-b))


def euclidean_distance_cdist(a, b):
    return cdist(a, b, metric='euclidean')


def main():
    print("Euclidean distance performance comparison")
    print("==========================================")
    print("Using timeit module: (times are in seconds)")
    print("--------------------")
    # print("euclidean_distance(a, b):", timeit.timeit("euclidean_distance(a, b)",
    #       setup="from __main__ import euclidean_distance; import numpy as np; a = np.array([1, 2]); b = np.array([3, 4])"))
    # print("euclidean_distance_np_linalg(a, b):", timeit.timeit("euclidean_distance_np_linalg(a, b)",
    #       setup="from __main__ import euclidean_distance_np_linalg; import numpy as np; a = np.array([1, 2]); b = np.array([3, 4])"))
    # print("euclidean_distance_np_sqrt(a, b):", timeit.timeit("euclidean_distance_np_sqrt(a, b)",
    #       setup="from __main__ import euclidean_distance_np_sqrt; import numpy as np; a = np.array([1, 2]); b = np.array([3, 4])"))
    # print("euclidean_distance_np_sqrt_square(a, b):", timeit.timeit("euclidean_distance_np_sqrt_square(a, b)",
    #       setup="from __main__ import euclidean_distance_np_sqrt_square; import numpy as np; a = np.array([1, 2]); b = np.array([3, 4])"))
    # print("euclidean_distance_np_dot(a, b):", timeit.timeit("euclidean_distance_np_dot(a, b)",
    #       setup="from __main__ import euclidean_distance_np_dot; import numpy as np; a = np.array([1, 2]); b = np.array([3, 4])"))
    # print("euclidean_distance_cdist(a, b):", timeit.timeit("euclidean_distance_cdist(a, b)",
    #       setup="from __main__ import euclidean_distance_cdist; import numpy as np; a = np.array([[1, 2], [1, 2]]); b = np.array([[3, 4], [3, 4]])"))

    ### Plotting ###
    # TODO matplotlib historgram of the times
    print("Plotting...")
    results = np.ones(shape=(6,))
    results[0] = timeit.timeit("euclidean_distance(a, b)",
                               setup="from __main__ import euclidean_distance; import numpy as np; a = np.array([1, 2]); b = np.array([3, 4])")
    results[1] = timeit.timeit("euclidean_distance_np_linalg(a, b)",
                               setup="from __main__ import euclidean_distance_np_linalg; import numpy as np; a = np.array([1, 2]); b = np.array([3, 4])")
    results[2] = timeit.timeit("euclidean_distance_np_sqrt(a, b)",
                               setup="from __main__ import euclidean_distance_np_sqrt; import numpy as np; a = np.array([1, 2]); b = np.array([3, 4])")
    results[3] = timeit.timeit("euclidean_distance_np_sqrt_square(a, b)",
                               setup="from __main__ import euclidean_distance_np_sqrt_square; import numpy as np; a = np.array([1, 2]); b = np.array([3, 4])")
    results[4] = timeit.timeit("euclidean_distance_np_dot(a, b)",
                               setup="from __main__ import euclidean_distance_np_dot; import numpy as np; a = np.array([1, 2]); b = np.array([3, 4])")
    results[5] = timeit.timeit("euclidean_distance_cdist(a, b)",
                               setup="from __main__ import euclidean_distance_cdist; import numpy as np; a = np.array([[1, 2], [1, 2]]); b = np.array([[3, 4], [3, 4]])")
    labels = ['own', 'np_linalg', 'np_sqrt',
              'np_sqrt_square', 'np_dot', 'scipy_cdist']
    yticks = np.arange(0, results.max(), 0.1)
    cmap = cm.get_cmap('plasma')
    norm = plt.Normalize(vmin=yticks.min(), vmax=yticks.max())
    plt.figure(figsize=(10, 5))
    plt.bar(x=range(0,6), height=results, color=cmap(norm(yticks)))
    plt.xticks(ticks=range(0,6), labels=labels, rotation=45)
    plt.yticks(ticks=yticks)
    plt.xlabel('Methods')
    plt.ylabel('Time (s)')
    plt.title('Euclidean distance performance comparison')
    plt.tight_layout()
    plt.savefig('/media/pecneb/970evoplus/gitclones/computer_vision_research/documents/TDK_v2/euclidean_distance_performance.png')


if __name__ == "__main__":
    main()
