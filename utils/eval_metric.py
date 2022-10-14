from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.metrics._ranking import _binary_clf_curve
from sklearn.exceptions import UndefinedMetricWarning
from skimage.measure import label
from sklearn.metrics import precision_recall_curve, roc_curve

import numpy as np
import warnings
import torch


def intersectionAndUnion(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target


def iou_metric(gt, pred):
    tn, fp, fn, tp = confusion_matrix(gt, pred).ravel()
    iou = tp / (tp + fp + fn)
    return iou


def pro_curve_iter(gt, scores, fpr=None, thresholds=None):
    '''
    calculate PRO-curve for evaluation,
    :param gt: ground truth of clf, 1-d numpy array binary mask [0,1]
    :param scores: clf scores
    :param fpr: fpr calculated by sklearn.metrics.roc_curve
    :param threshold: threshold calculated by sklearn.metrics.roc_curve
    :return: fpr, pro, thresholds
    '''

    assert gt.shape == scores.shape
    if fpr == None or thresholds == None:
        fpr, _, thresholds = roc_curve(gt, scores)
    assert fpr.shape == thresholds.shape

    # PRO curve up to an average false-positive rate of 30%
    index_range = np.where(fpr <= 0.3)
    index_range = index_range[0]
    fpr = fpr[index_range]
    thresholds = thresholds[index_range]

    pro = []
    # calculate per region overlap for each threshold
    for threshold in thresholds:
        mask = np.copy(scores)
        mask[mask >= threshold] = 1
        mask[mask < threshold] = 0
        iou = iou_metric(gt, mask)
        print(iou)
        pro.append(iou)
    pro = np.array(pro)
    assert pro.shape == fpr.shape

    return fpr, pro, thresholds


def iou_curve(gt, scores, pos_label=None, sample_weight=None,
              drop_intermediate=True):
    '''
    Compute per region overlap-false positive rate pairs for different probability thresholds
    exceeded from sklearn
    ---------
    :param gt: ground truth of clf, 1-d numpy array binary mask [0,1]
    :param scores: clf scores
    :return: fpr, pro, thresholds
    '''

    assert gt.shape == scores.shape

    fps, tps, thresholds = _binary_clf_curve(
        gt, scores, pos_label=pos_label, sample_weight=sample_weight)

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(tps, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if tps.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        tps = np.r_[0, tps]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    if tps[-1] <= 0:
        warnings.warn("No positive samples in y_true, "
                      "true positive value should be meaningless",
                      UndefinedMetricWarning)
        tpr = np.repeat(np.nan, tps.shape)
    else:
        tpr = tps / tps[-1]


    # tns = fps[-1] - fps
    fns = tps[-1] - tps
    # compute per-region overlap from confusion-matrix
    iou = tps / (tps + fps + fns)

    # PRO curve up to an average false-positive rate of 30%
    index_range = np.where(fpr <= 0.3)
    index_range = index_range[0]
    fpr = fpr[index_range]
    iou = iou[index_range]
    thresholds = thresholds[index_range]

    return fpr, iou, thresholds


# def per_region_overlap(gt, score)


def pro_curve(gt, scores, pos_label=None, sample_weight=None,
              drop_intermediate=True):
    '''
    Compute per region overlap-false positive rate pairs for different probability thresholds
    exceeded from sklearn
    ---------
    :param gt: ground truth of clf, numpy array binary mask [0,1], shape (N x H x W)
    :param scores: clf scores, normalized numpy array, shape (N x H x W)
    :return: fpr, pro, thresholds
    '''

    assert gt.shape == scores.shape
    regions = np.zeros_like(gt)        # connected component in Ground Truth
    # detections = []     # corresponding scores in regions

    counter = 0     # count the number of connected regions in test dataset

    N = gt.shape[0]     # N images, both normal and anomaly
    for n in range(N):

        # select each binary mask and predicted score map of (h x w)
        mask = gt[n]
        # pred = scores[n]
        # compute connected regions if gt is anomaly
        if mask.__contains__(1):
            labels, num = label(mask, connectivity = 2, return_num=True)
            index_pos = np.where(labels != 0)
            labels[index_pos] = labels[index_pos] + counter
            regions[n] = labels.copy()
            counter = counter + num

    assert gt.shape == regions.shape
    # compute per region overlap in each region
    regions = regions.flatten()
    scores = scores.flatten()

    fps, tps, thresholds = _binary_clf_curve(
        gt.flatten(), scores, pos_label=pos_label, sample_weight=sample_weight)
    pro = np.zeros_like(thresholds)

    # for each connected region of anomaly
    # overlap = tp / (tp + fn), also named as TPR
    for c in range(counter):
        # select one connected region, set to 1, others 0
        region_idx = np.where(regions == c + 1)
        region = np.zeros_like(scores)
        region[region_idx] = 1

        # compute region overlap
        per_fps, per_tps, per_thresholds = _binary_clf_curve(
            region, scores, pos_label=pos_label, sample_weight=sample_weight)

        assert (thresholds == per_thresholds).all()

        if per_tps[-1] <= 0:
            warnings.warn("No positive samples in y_true, "
                          "true positive value should be meaningless",
                          UndefinedMetricWarning)
            overlap = np.repeat(np.nan, tps.shape)
        else:
            overlap = per_tps / per_tps[-1]

        pro = pro + overlap

    pro = pro / counter     # the average coverage across all regions
    # combine fpr, pro, thresholds

    # Attempt to drop thresholds corresponding to points in between and
    # collinear with other points. These are always suboptimal and do not
    # appear on a plotted ROC curve (and thus do not affect the AUC).
    # Here np.diff(_, 2) is used as a "second derivative" to tell if there
    # is a corner at the point. Both fps and tps must be tested to handle
    # thresholds with multiple data points (which are combined in
    # _binary_clf_curve). This keeps all cases where the point should be kept,
    # but does not drop more complicated cases like fps = [1, 3, 7],
    # tps = [1, 2, 4]; there is no harm in keeping too many thresholds.
    if drop_intermediate and len(fps) > 2:
        optimal_idxs = np.where(np.r_[True,
                                      np.logical_or(np.diff(fps, 2),
                                                    np.diff(pro, 2)),
                                      True])[0]
        fps = fps[optimal_idxs]
        pro = pro[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    if pro.size == 0 or fps[0] != 0:
        # Add an extra threshold position if necessary
        pro = np.r_[0, pro]
        fps = np.r_[0, fps]
        thresholds = np.r_[thresholds[0] + 1, thresholds]

    if fps[-1] <= 0:
        warnings.warn("No negative samples in y_true, "
                      "false positive value should be meaningless",
                      UndefinedMetricWarning)
        fpr = np.repeat(np.nan, fps.shape)
    else:
        fpr = fps / fps[-1]

    # if tps[-1] <= 0:
    #     warnings.warn("No positive samples in y_true, "
    #                   "true positive value should be meaningless",
    #                   UndefinedMetricWarning)
    #     tpr = np.repeat(np.nan, tps.shape)
    # else:
    #     tpr = tps / tps[-1]


    # PRO curve up to an average false-positive rate of 30%
    index_range = np.where(fpr <= 0.3)
    index_range = index_range[0]
    fpr = fpr[index_range]
    # tpr = tpr[index_range]
    pro = pro[index_range]
    thresholds = thresholds[index_range]

    return fpr, pro, thresholds


def PRO_score(fpr, pro):
    '''
    calculate PRO score using fpr and pro curve
    :param fpr: numpy array 0 - 0.3
    :param pro: numpy array
    :return:
    '''
    assert fpr.shape == pro.shape
    assert fpr.max() <= 0.3
    full = np.ones_like(pro)
    auc_pro = auc(fpr, pro)
    auc_full = auc(fpr, full)
    pro_score = auc_pro / auc_full

    return pro_score


def fpfn(gt, scores, threshold):
    """
    quantity evaluation for fp and fn
    :param gt: ground truth
    :param scores: predicted anomaly score map
    :param threshold: corresponding threshold to calculate binary map
    :return:
    """
    assert gt.shape == scores.shape
    mask = scores

    # get segmentation mask
    mask[mask <= threshold] = 0
    mask[mask > threshold] = 1

    tn, fp, fn, tp = confusion_matrix(gt.flatten(), mask.flatten()).ravel()

    print("tn:{} || fp:{} || fn:{} || tp:{} ||".format(tn, fp, fn, tp))
    fpr = fp / (fp + tn)
    fnr = fn / (fn + tp)

    return fp, fn, fpr, fnr


def analyze_region(gt, scores, threshold):
    """
    by dlluo
    analyzing region-based fn & fp, counting the number of regions
    :param gt: ground truth
    :param scores: predicted anomaly score map
    :param threshold: corresponding threshold to calculate binary map
    :return: []
    """
    assert gt.shape ==scores.shape
    masks = scores

    # initializing
    # gt_regions = np.zeros_like(gt)  # connected component in Ground Truth
    # mask_regions = np.zeros_like(masks)

    # get segmentation mask
    masks[masks <= threshold] = 0
    masks[masks > threshold] = 1
    # print(masks.max())

    N = gt.shape[0]  # N images, both normal and anomaly
    result = [] # [num_gt_region, num_mask_region, fn, fp]

    # counter = 0 # count the number of connected regions in test dataset

    escape_area = []
    overkill_area = []

    # count connected-component in each image
    for n in range(N):
        # select each binary mask and predicted score map of (h x w)
        gt_mask = gt[n]
        mask = masks[n]
        # pred = scores[n]
        # compute connected regions if gt is anomaly
        hit = 0
        cover = 0
        if gt_mask.__contains__(1):
            labels_gt, num_gt = label(gt_mask, connectivity = 2, return_num=True)
            # index_pos = np.where(labels_gt != 0)
            # labels[index_pos] = labels[index_pos] + counter
            # gt_regions[n] = labels.copy()
            # if a gt_region has overlap with predicted mask, it is hit
            for i in range(num_gt):
                region_idx = np.where(labels_gt == i + 1)
                region = np.zeros_like(gt_mask)
                region[region_idx] = 1
                div = region - mask
                if div[region_idx].__contains__(0):
                    hit = hit + 1
                else:
                    escape_area.append(len(region[region_idx]))

        else:
            num_gt = 0

        mis_hit = num_gt - hit

        if mask.__contains__(1):
            labels_mask, num_mask = label(mask, connectivity=2, return_num=True)

            # if a predicted region does not any gt region, it is fp
            for i in range(num_mask):
                region_idx = np.where(labels_mask == i + 1)
                region = np.zeros_like(mask)
                region[region_idx] = 1
                div = region - gt_mask
                if div[region_idx].__contains__(0):
                    cover = cover + 1
                else:
                    overkill_area.append(len(region[region_idx]))

        else:
            num_mask = 0

        num_fp = num_mask - cover

        result.append([num_gt, num_mask, hit, mis_hit, cover, num_fp])
    sum_result = np.sum(result, axis=0)
    print("totally {} GT_regions and {} mask_regions, escape:{}, overkill:{}".format(repr(sum_result[0]),repr(sum_result[1]),repr(sum_result[3]),repr(sum_result[5])))
    if len(escape_area) != 0:
        print("escape region area, max:{}pix, min:{}pix ".format(repr(np.max(escape_area)), repr(np.min(escape_area))))
    if len(overkill_area) != 0:
        print("overkill region area, max:{}pix, min:{}pix ".format(repr(np.max(overkill_area)), repr(np.min(overkill_area))))

    return result, escape_area, overkill_area


def analyze_area(gt, scores, threshold):

    assert gt.shape ==scores.shape
    masks = scores
    N = gt.shape[0]  # N images, both normal and anomaly

    # get segmentation mask
    masks[masks <= threshold] = 0
    masks[masks > threshold] = 1

    ious = []
    fpfn = []

    tn, fp, fn, tp = confusion_matrix(gt.flatten(), masks.flatten()).ravel()
    iou = tp / (tp + fp + fn)

    ious.append(iou)
    fpfn.append([fp, fn])

    sum_fpfn = np.sum(fpfn, axis=0)

    print("total escape:{}, total overkill:{}".format(repr(sum_fpfn[1]), repr(sum_fpfn[0])))

    return ious, fpfn

def get_aupr_curve(gt_list, scores):
    """
    by dlluo
    :param gt_list: ground truth
    :param scores: predicted anomaly score map
    :return: precision, recall, thresholds
    """
    assert gt_list.shape == scores.shape
    precision, recall, thresholds = precision_recall_curve(gt_list.flatten(), scores.flatten())
    a       = 2 * precision * recall
    b       = precision + recall
    f1      = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    best_th = thresholds[np.argmax(f1)]
    aupr    = auc(recall, precision)
    return aupr, best_th, precision, recall