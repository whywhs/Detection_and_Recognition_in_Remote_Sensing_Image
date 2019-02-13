import numpy as np

from cpu_nms import cpu_nms, cpu_soft_nms
from gpu_nms import gpu_nms
import polyiou

def soft_nms(dets, sigma=0.6, Nt=0.3, threshold=0.001, method=1):
    keep = cpu_soft_nms(np.ascontiguousarray(dets, dtype=np.float32),
                        np.float32(sigma), np.float32(Nt),
                        np.float32(threshold),
                        np.uint8(method))
    return keep

def py_nms_wrapper(thresh):
    def _nms(dets):
        return nms(dets, thresh)
    return _nms


def cpu_nms_wrapper(thresh):
    def _nms(dets):
        return cpu_nms(dets, thresh)
    return _nms


def gpu_nms_wrapper(thresh, device_id):
    def _nms(dets):
        return gpu_nms(dets, thresh, device_id)
    return _nms


def nms(dets, thresh):
    """
    greedily select boxes with high confidence and overlap with current maximum <= thresh
    rule out overlap >= thresh
    :param dets: [[x1, y1, x2, y2 score]]
    :param thresh: retain overlap < thresh
    :return: indexes to keep
    """
    '''
    if dets.shape[0] == 0:
        return []

    #x1 = dets[:, 0]
    #y1 = dets[:, 1]
    #x2 = dets[:, 2]
    #y2 = dets[:, 3]

    ex_x = np.vstack((dets[:, 0], dets[:, 2], dets[:, 4], dets[:, 6]))
    ex_y = np.vstack((dets[:, 1], dets[:, 3], dets[:, 5], dets[:, 7]))
    x1 = np.amin(ex_x, axis=0)
    y1 = np.amin(ex_y, axis=0)
    x2 = np.amax(ex_x, axis=0)
    y2 = np.amax(ex_y, axis=0)


    scores = dets[:, 8]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep
    '''
    if dets.shape[0] == 0:
        return []

    scores = dets[:, 8]
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        for j in range(order.size - 1):
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)
        #print 'ovr is',ovr
        inds = np.where(ovr <= thresh)[0]
        #print 'inds is',inds
        order = order[inds + 1]
        #print 'order is',order
        
    return keep