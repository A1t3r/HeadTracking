import numpy as np
import cv2
from lap import lapjv


def iou_vector(bbox, candidates):
    """Computer intersection over union.
    Parameters
    ----------
    bbox : ndarray
        A bounding box in format `(top left x, top left y, width, height)`.
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.
    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.
    """
    bbox_tl, bbox_br = bbox[:2], bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return 1 - area_intersection / (area_bbox + area_candidates - area_intersection)


def iou(detA, detB):
    xA = max(detA.coord[0], detB.coord[0])
    yA = max(detA.coord[1], detB.coord[1])
    xB = min(detA.coord[0] + detA.l, detB.coord[0] + detB.l)
    yB = min(detA.coord[1] + detA.h, detB.coord[1] + detB.h)
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA) * max(0, yB - yA)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = detA.h * detA.l
    boxBArea = detB.h * detB.l
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def calcutale_Bhattacharyya_distance(tracks, detections):
    hist_sim = np.eye(len(tracks), len(detections))
    for i in range(len(tracks)):
        for j in range(len(detections)):
            hist_sim[i][j] = cv2.compareHist(tracks[i].color_hist, detections[j].color_hist,
                                             cv2.HISTCMP_BHATTACHARYYA)
            if hist_sim[i][j] > 0.2: hist_sim[i][j] = 1.
    return hist_sim


def compute_new_hsv(im):
    """
    Illuminance and Gamma invariant HSV
    """
    eps = 1e-10
    r, g, b = np.array(cv2.split(im)) + eps
    traditional_hsv = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)
    numerator = np.log(r) - np.log(g)
    denominator = np.log(r) + np.log(g) - 2 * np.log(b) + eps
    new_hue = np.clip(np.round(numerator / denominator).astype(np.uint8), 0, 180)
    new_hsv = np.zeros_like(traditional_hsv).astype(np.uint8)
    new_hsv[:, :, 0] = new_hue
    new_hsv[:, :, 1] = traditional_hsv[:, :, 1]
    new_hsv[:, :, 2] = traditional_hsv[:, :, 2]
    return new_hsv


def compute_histogram(im, kernel=True):
    """
    im : cropped patch
    kernel : Will perform circular masking
    """
    x, y = im.shape[:2]
    # if kernel:
    #     mask = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (y, x))
    #     mask = mask[:, :, np.newaxis]
    # else:
    #     mask = np.ones((x, y, 1)).astype(np.uint8)
    hsv_im = compute_new_hsv(im)
    channels = [0, 1, 2]
    hist_size = [8, 8, 8]
    hist_range = [0, 180, 0, 256, 0, 256]
    # hist_hue = cv2.calcHist([hsv_im], [0],
    #                         mask, [30], [0, 180], False)
    # cv2.normalize(hist_hue,hist_hue,0,255,cv2.NORM_MINMAX)
    hist = cv2.calcHist([hsv_im], channels,
                        None, hist_size, hist_range)
    image_hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
    return image_hist


def add_expensive_edges(costs):
    """Replaces non-edge costs (nan, inf) with large number.
    If the optimal solution includes one of these edges,
    then the original problem was infeasible.
    Parameters
    ----------
    costs : np.ndarray
    """
    # The graph is probably already dense if we are doing this.
    assert isinstance(costs, np.ndarray)
    # The linear_sum_assignment function in scipy does not support missing edges.
    # Replace nan with a large constant that ensures it is not chosen.
    # If it is chosen, that means the problem was infeasible.
    valid = np.isfinite(costs)
    if valid.all():
        return costs.copy()
    if not valid.any():
        return np.zeros_like(costs)
    r = min(costs.shape)
    # Assume all edges costs are within [-c, c], c >= 0.
    # The cost of an invalid edge must be such that...
    # choosing this edge once and the best-possible edge (r - 1) times
    # is worse than choosing the worst-possible edge r times.
    # l + (r - 1) (-c) > r c
    # l > r c + (r - 1) c
    # l > (2 r - 1) c
    # Choose l = 2 r c + 1 > (2 r - 1) c.
    c = np.abs(costs[valid]).max() + 1  # Doesn't hurt to add 1 here.
    large_constant = 2 * r * c + 1
    return np.where(valid, costs, large_constant)


def lsa_solve_lapjv(costs):
    """Solves the LSA problem using lap.lapjv()."""

    # The lap.lapjv function supports +inf edges but there are some issues.
    # https://github.com/gatagat/lap/issues/20
    # Therefore, replace nans with large finite cost.
    finite_costs = add_expensive_edges(costs)
    row_to_col, _ = lapjv(finite_costs, return_cost=False, extend_cost=True)
    indices = np.array([np.arange(costs.shape[0]), row_to_col], dtype=int).T
    # Exclude unmatched rows (in case of unbalanced problem).
    indices = indices[indices[:, 1] != -1]  # pylint: disable=unsubscriptable-object
    rids, cids = indices[:, 0], indices[:, 1]
    # Ensure that no missing edges were chosen.
    rids, cids = _exclude_missing_edges(costs, rids, cids)
    return rids, cids


def _exclude_missing_edges(costs, rids, cids):
    subset = [
        index for index, (i, j) in enumerate(zip(rids, cids))
        if np.isfinite(costs[i, j])
    ]
    return rids[subset], cids[subset]
