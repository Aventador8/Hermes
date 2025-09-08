from abc import abstractmethod
import numpy as np
from torch.nn import CrossEntropyLoss
from Code.loss.pixel_contrast import *

def normalize(x, axis=-1):
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x
def ce_dice_loss(preds,target):
    targets = target.squeeze(dim=1).long()

    ce_loss = CrossEntropyLoss_(weight=0.5)
    # dice_loss = DiceLoss_(weight=0.5, n_classes=2, class_weight=[0.0, 1.0])
    dice_loss = DiceLoss_(weight=1.0, n_classes=2, class_weight=[0.0, 1.0])

    # print('preds type: ', type(preds))
    # print('targets type: ', type(targets))

    ce = ce_loss(preds, targets)
    dice = dice_loss(preds, targets)

    # ce_loss = F.cross_entropy(preds,targets)
    # dice = DiceLoss(mode='multiclass', ignore_index=0)
    # # preds = torch.argmax(preds, dim=1)
    # dice_loss = dice(preds.to(float),targets.to(torch.int64))
    loss = (ce + dice) / 2.0
    return loss, ce, dice





def segAndclass_consistency(intra_cls,intra_seg):

    emb_1 = F.normalize(intra_cls, dim=1)
    emb_2 = F.normalize(intra_seg, dim=1)

    return 1 - F.cosine_similarity(emb_1, emb_2, dim=-1).mean()



def consistency_loss(Segweak,SegStrong):
    weak = Segweak.detach()
    strong = SegStrong
    weak = F.softmax(weak, dim=1) / 0.5
    strong = F.softmax(strong, dim=1)
    weak = weak.view(weak.shape[0],-1)
    strong = strong.view(strong.shape[0],-1)

    loss = 1 - F.cosine_similarity(weak,strong,dim=-1)

    return loss.mean()


def calc_unsupervised_loss(p1, p2, z1, z2,
                           seg_weak_out,certainty_pseudo,
                           seg_strong_feature,seg_strong_out,
                           intra_task_cls_consistency,intra_task_seg_consistency,
                           device, step,
                           contrast_weight = 0.3, consitent_weight = 0.3):
    # print('seg_weak_out',seg_weak_out.size())

    # classsification  loss
    # imageContrastLoss = ImageContrastLoss(batch_size)
    # class_contrast = imageContrastLoss(class_weak,class_strong)


    if step > 1000:

        # contrast   segmentation  loss
        pixelContrastLoss = PixelContrastLoss().to(device)
        probsStrong = torch.softmax(seg_strong_out, dim=1)
        _,predictStrong = torch.max(probsStrong, dim=1)
        seg_contrast = pixelContrastLoss(seg_strong_feature, certainty_pseudo, predictStrong)
    # print('probsWeak',probsWeak.size())
    # print('psuedoLabels',psuedoLabels.size())
    # print('mask',mask.size())
    # print('probsStrong',probsStrong.size())
    # print('predictStrong',predictStrong.size())

    else:
        seg_contrast = torch.tensor(0.0).to(device)

    # class_out: N * 128  seg_feature: N * 16 * 512 * 512  outSeg: N * n_class * 512 * 512


    seg_consistency = consistency_loss(seg_weak_out,seg_strong_out)
    task_consistency = segAndclass_consistency(intra_task_cls_consistency,intra_task_seg_consistency)


    # total loss
    # print('class_contrast loss',cls_contrast)
    # print('seg_contrast loss',seg_contrast)
    # print('seg_consistency loss',seg_consistency)
    # print('task_consistency loss', task_consistency)
    loss = contrast_weight * ( seg_contrast) + consitent_weight * (seg_consistency + task_consistency)

    return loss, seg_contrast, seg_consistency, task_consistency



def calc_supervised_loss(out_seg,mask,class_preds=None,label=None):


    # segmentation loss   loss, ce_loss, dice_loss
    seg_total_loss, seg_ce_loss, seg_dice_loss = ce_dice_loss(out_seg,mask)

    # classification loss
    loss_criterion = torch.nn.CrossEntropyLoss()

    if class_preds is not None and label is not None:
        class_loss = loss_criterion(class_preds,label)
        loss = seg_total_loss + class_loss
        return {'total_loss': loss, 'seg_dice_loss': seg_dice_loss,
                'seg_ce_loss': seg_ce_loss, 'seg_total_loss': seg_total_loss, 'cls_loss': class_loss}
    else:
        loss = seg_total_loss

        return {'total_loss': loss, 'seg_dice_loss': seg_dice_loss,
                'seg_ce_loss': seg_ce_loss,'seg_total_loss': seg_total_loss}



def calc_supervised_loss_cls_branch(class_preds,label,out_seg=None,mask=None):


    # classification loss
    loss_criterion = torch.nn.CrossEntropyLoss()
    class_loss = loss_criterion(class_preds, label)

    if out_seg is not None and mask is not None:
    # segmentation loss   loss, ce_loss, dice_loss
        seg_total_loss, seg_ce_loss, seg_dice_loss = ce_dice_loss(out_seg,mask)
        loss = seg_total_loss + class_loss
        return {'total_loss': loss, 'seg_dice_loss': seg_dice_loss,
                'seg_ce_loss': seg_ce_loss, 'seg_total_loss': seg_total_loss, 'cls_loss': class_loss}
    else:
        loss = class_loss

        return {'total_loss': loss, 'cls_loss': class_loss}


# supervised loss
# intraTask_loss = -(torch.log(torch.mean(trace)))

def make_one_hot(input, num_classes):
    """Convert class index tensor to one hot encoding tensor.
    Args:
         input: A tensor of shape [N, 1, *]
         num_classes: An int of number of class
    Returns:
        A tensor of shape [N, num_classes, *]
    """
    shape = np.array(input.shape)
    shape[1] = num_classes
    shape = tuple(shape)
    result = torch.zeros(shape)
    result = result.scatter_(1, input.cpu(), 1)

    return result


class BaseWeightLoss:
    def __init__(self, name='loss', weight=1.) -> None:
        super().__init__()
        self.name = name
        self.weight = weight

    @abstractmethod
    def _cal_loss(self, preds, targets, **kwargs):
        pass

    def __call__(self, preds, targets, **kwargs):
        return self._cal_loss(preds, targets, **kwargs) * self.weight


class CrossEntropyLoss_(BaseWeightLoss):
    def __init__(self, name='loss_ce', weight=1., **kwargs) -> None:
        super().__init__(name, weight)
        self.loss = CrossEntropyLoss(**kwargs)

    def _cal_loss(self, preds, targets, **kwargs):
        targets = targets.to(torch.long).squeeze(1)
        return self.loss(preds, targets)



class DiceLoss_(BaseWeightLoss):
    def __init__(self, name='loss_dice', weight=1., smooth=1e-5, n_classes=2, class_weight=None, softmax=True,
                 **kwargs):
        super().__init__(name, weight)
        self.n_classes = n_classes
        self.smooth = smooth
        self.class_weight = [1.] * self.n_classes if class_weight is None else class_weight
        self.softmax = softmax

    def _one_hot_encoder(self, targets):
        target_list = []
        for _ in range(self.n_classes):
            temp_prob = targets == _ * torch.ones_like(targets)
            target_list.append(temp_prob)
        output_tensor = torch.cat(target_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, pred, target, ignore):
        assert pred.shape[0] == target.shape[0]
        intersect = torch.sum(torch.mul(pred[ignore != 1], target[ignore !=1]))
        loss = 1 - (2 * intersect + self.smooth) / (torch.sum(pred[ignore !=1].pow(2)) + torch.sum(target[ignore !=1].pow(2)) + self.smooth)
        return loss

    def _cal_loss(self, preds, targets, ignore=None, **kwargs):
        if self.softmax:
            preds = torch.softmax(preds, dim=1)
        targets = self._one_hot_encoder(targets.unsqueeze(1))
        # print('preds shape: ', preds.shape)
        # print('targets shape: ', targets.shape)
        assert preds.size() == targets.size(), 'pred & target shape do not match'
        loss = 0.0
        for _ in range(self.n_classes):
            dice = self._dice_loss(preds[:, _], targets[:, _], ignore)
            loss += dice * self.class_weight[_]
        return loss / self.n_classes






if __name__ == "__main__":
    x1 = torch.rand([4, 3])
    print(x1)
    a = x1.unsqueeze(1)
    print(a.shape)









