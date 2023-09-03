# first line: 296
@memory.cache
def search_min_max_coordinates(trackedObjects: list[TrackedObject]):
    max_y = 0 
    min_y = 9999 
    max_x = 0
    min_x = 9999
    coord = np.array([]).reshape((0,2))
    X = np.array([])
    Y = np.array([])
    for obj in tqdm.tqdm(trackedObjects, desc="Looking for min max values."):
        X = np.append(X, obj.history_X)
        Y = np.append(Y, obj.history_Y)
    min_x = np.min(X)
    max_x = np.max(X)
    min_y = np.min(Y)
    max_y = np.max(Y)
    return min_x, min_y, max_x, max_y
