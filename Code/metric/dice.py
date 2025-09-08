import torch
import torch.nn.functional as F


def diceCoeff(pred, gt, smooth=1e-5):
    r""" computational formula：
        dice = (2 * (pred ∩ gt)) / (pred ∪ gt)
    """
    pred = pred.log_softmax(dim=1).exp()

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = (2 * intersection + smooth) / (unionset + smooth)

    return loss.sum() / N


class Dice:
    def __init__(self, num_classes):
        super(Dice, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):  # y_pred: N,C,H,W      y_true: N,H,W
        num_classes = y_pred.size(1)
        y_true = F.one_hot(y_true, num_classes)  # N,H,W -> N,H,W, C
        y_true = y_true.permute(0, 3, 2, 1)  # N, C, H, W
        class_dice = []
        for i in range(self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :, :], y_true[:, i:i + 1, :, :]))
        mean_dice = sum(class_dice) / len(class_dice)
        return mean_dice, class_dice




def dice_coef(output, target):
    output = torch.argmax(torch.softmax(output, dim=1), dim=1)
    smooth = 1e-5
    output = output.view(-1).data.cpu().numpy()
    target = target.view(-1).data.cpu().numpy()
    intersection = (output * target).sum()

    return (2. * intersection + smooth) / (output.sum() + target.sum() + smooth)


#  计算多类别dice系数
def calc_dice_score(dice_class, pred, mask, nclass=3, smooth=1e-5):  # pred: N, H ,W          mask: N, H, W
    pred = torch.argmax(torch.softmax(pred, dim=1), dim=1)
    mask = mask.squeeze(1)
    # print('pred.shape', pred.shape)
    # print('mask.shape', mask.shape)
    assert pred.shape == mask.shape
    for cls in range(nclass):
        inter = ((pred == cls) * (mask == cls)).sum().item()
        union = (pred == cls).sum().item() + (mask == cls).sum().item()
        dice_class[cls] += (2.0 * inter + smooth ) / ( union + smooth)


def ab(dice, x , y):
    for i in range(0,3):
        dice[i] = x+y+i



if __name__ == "__main__":
    # pred = torch.rand((1,3,8,8))
    # mask = torch
    dice = [0]*3
    ab(dice,2,2)
    print(dice)







