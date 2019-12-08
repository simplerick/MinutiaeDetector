import numpy as np
import cv2


def find_nearest_point(x,y,points,index=False):
    """
    Finds point (x_,y_) from points closest to (x,y)
    and returns distance, x_ and y_.
    """
    dst = 99999999
    idx = -1
    for i,(x_0,y_0) in enumerate(points):
        dist = max(abs(x_0 - x),abs(y_0 - y))
        if dist < dst:
            x_, y_ = x_0, y_0
            dst = dist
            idx = i
    if index:
        return dst, x_, y_, idx
    return dst, x_, y_


def get_coord(vec,size,inter_dist):
    """
    Converts a vector of two real values (a, b) to the coordinates of point (x, y).
    (a,b) = [(x-x_c)/inter_dist, (y-y_c)/inter_dist], if a,b are in range from -1 to 1,
    then (x,y) - some point inside the central square with side 2*inter_dist
    """
    center = size[0]//2, size[1]//2
    return (vec*inter_dist+center).round().astype('int')



def compute_image_relative_coord(coord, total_stride):
    grid = np.indices(coord.shape[:-1]).transpose(1,2,0)
    return coord + total_stride*grid



def grid(array, ncols=8):
    """
    Makes grid from batch of images with shape (n_batch, height, width, channels)
    """
    array = np.pad(array, [(0,0),(1,1),(1,1),(0,0)], 'constant')
    nindex, height, width, intensity = array.shape
    nrows = (nindex+ncols-1)//ncols
    r = nrows*ncols - nindex # remainder
    # want result.shape = (height*nrows, width*ncols, intensity)
    arr = np.concatenate([array]+[np.zeros([1,height,width,intensity])]*r)
    print(arr.shape)
    result = (arr.reshape(nrows, ncols, height, width, intensity)
              .swapaxes(1,2)
              .reshape(height*nrows, width*ncols, intensity))
    return np.pad(result, [(1,1),(1,1),(0,0)], 'constant')




def add_points(image, points,p=1.,rgb_color=(1,0,0), d=8):
    """
    Add rectangles around points to the image with given opacity
    """
    if isinstance(p, np.ndarray):
        # maybe there are faster ways to draw multiple rectangles
        for (x, y),prob in zip(points,p):
            overlay = image.copy()
            cv2.rectangle(overlay, (int(x)-d, int(y)-d), (int(x)+d, int(y)+d), rgb_color, 1)
            image = cv2.addWeighted(overlay,prob,image,1-prob,0)
        return image
    else:
        overlay = image.copy()
        for (x, y) in points:
            cv2.rectangle(overlay, (int(x)-d, int(y)-d), (int(x)+d, int(y)+d), rgb_color, 1)
        return cv2.addWeighted(overlay,p,image,1-p,0)




def match_points(points, points_pred, prob, tol=4):
    """
    Matches predicted points with ground truth points.
    Prediction is considered correct if there is ground truth point
    within a distance = tol.
    """
    target = []
    pred = []
    for x1,y1 in points:
        dist, x2, y2, i = find_nearest_point(x1,y1,points_pred, index=True)
        target.append(1)
        if dist <= tol:
            pred.append(prob[i])
            points_pred[i] = [-1,-1]
        else:
            pred.append(0)
    for i, (x,y) in enumerate(points_pred):
        if (x,y) != [-1,-1]:
            target.append(0)
            pred.append(prob[i])
    return target, pred
