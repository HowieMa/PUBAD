import torch
import torch.nn.functional as F


def regr_loss(regr, gt_regr, mask):
    """
    :param regr:        Tensor  size:(B,C,H,W)  C=num pose
    :param gt_regr:     Tensor  size:(B,C,H,W)  C=num pose
    :param mask:        Tensor  size:(B,1,H,W)  C=num pose
    :return:
    """

    mask = mask.expand(regr.shape)

    num = (mask == 1).float().sum() * 2

    regr = regr[mask == 1]
    gt_regr = gt_regr[mask == 1]
    regr_loss = F.l1_loss(
        regr, gt_regr, size_average=False)
    regr_loss = regr_loss / (num + 1e-4)
    return regr_loss


def _neg_loss(pred, gt, alpha=2, beta=4):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
    Arguments:
        pred (batch x h x w)
        gt_regr (batch x h x w)
    '''
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, beta)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss


def centernet_loss(prediction, gt_mask, gt_reg):
    """
    :param prediction:  Tensor   size:(B,8,H,W)
    :param gt_mask:        Tensor   size:(B,H,W)
    :param gt_reg:        Tensor   size:(B,7,H,W)
    :return:
    """
    # Focal Loss for Binary keypoint mask
    mask_loss = _neg_loss(torch.sigmoid(prediction[:, 0, ...]), gt_mask)

    # Smooth L1 loss for obj pose
    pose_loss = regr_loss(prediction[:, 1:, :, :], gt_reg, gt_mask.unsqueeze(1))

    return mask_loss, pose_loss