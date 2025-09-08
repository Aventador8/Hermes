import numpy as np
import torch
import logging
import matplotlib.pyplot as plt
from torch import Tensor
import torch.nn.functional as F
from Code.metric.iou import intersectionAndUnion


def mask_to_onehot(mask, palette):
    """
    Converts a segmentation mask (H, W, C) to (H, W, K) where the last dim is a one
    hot encoding vector, C is usually 1 or 3, and K is the number of class.
    """
    semantic_map = []
    for colour in palette:
        equality = np.equal(mask, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1).astype(np.float32)
    return semantic_map


def onehot_to_mask(mask, palette):
    """
    Converts a mask (H, W, K) to (H, W, C)
    """
    x = np.argmax(mask, axis=-1)
    colour_codes = np.array(palette)
    x = np.uint8(colour_codes[x.astype(np.uint8)])
    return x

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, length=0):
        self.length = length
        self.reset()

    def reset(self):
        if self.length > 0:
            self.history = []
        else:
            self.count = 0
            self.sum = 0.0
        self.val = 0.0
        self.avg = 0.0

    def update(self, val, num=1):
        if self.length > 0:
            # currently assert num==1 to avoid bad usage, refine when there are some explict requirements
            assert num == 1
            self.history.append(val)
            if len(self.history) > self.length:
                del self.history[0]

            self.val = self.history[-1]
            self.avg = np.mean(self.history)
        else:
            self.val = val
            self.sum += val * num
            self.count += num
            self.avg = self.sum / self.count


def evaluate_iou(pred, mask, nclass=3):  # pred: N, H ,W      mask: N, H, W
    with torch.no_grad():
        pred = pred.argmax(dim=1)
        mask = mask.squeeze(1)
        intersection, union, target = \
            intersectionAndUnion(pred.cpu().numpy(), mask.cpu().numpy(), nclass, 255)

    return intersection, union, target



def add_figures(writer, saliency, step, padding = 2):   #
    fig, ax = plt.subplots(2, 4, figsize=(10, 10))
    for i, axi in enumerate(ax.flat):
        image = saliency[i].cpu().numpy()
        axi.imshow(image, cmap=plt.cm.hot)
        axi.set(xticks=[], yticks=[])
    writer.add_figure('train/seg/saliency', fig, step)


def tensorboard_write_scalars(writer, ordered_dict, step):
    for key, value in ordered_dict.items():
        if isinstance(value, Tensor):
            ordered_dict[key] = value.item()
        writer.add_scalar(key, value, step)

def tensorboard_write_images(writer, images_dict, step):
    for key, value in images_dict.items():
        writer.add_image(key, value, step)



logs = set()

def init_log(filename, name, level=logging.INFO):
    if (name, level) in logs:
        return
    logs.add((name, level))
    logger = logging.getLogger(name)
    logger.setLevel(level)

    format_str = "[%(asctime)s][%(levelname)8s] %(message)s"
    formatter = logging.Formatter(format_str)

    fh = logging.FileHandler(filename, 'w')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(level)
    # if "SLURM_PROCID" in os.environ:
    #     rank = int(os.environ["SLURM_PROCID"])
    #     logger.addFilter(lambda record: rank == 0)
    # else:
    #     rank = 0

    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def get_classwise_thresholds(pseudo_labels, num_classes, base_threshold=0.95, min_threshold=0.5):


    class_counts = torch.bincount(pseudo_labels.flatten(), minlength=num_classes)


    total_count = class_counts.sum().item()
    class_ratios = class_counts.float() / total_count


    classwise_thresholds = base_threshold - (base_threshold - min_threshold) * (1 - class_ratios)

    return classwise_thresholds


def filter_pseudo_labels_by_threshold(probabilities, pseudo_labels, classwise_thresholds):

    # print('probabilities shape:', probabilities.shape)
    # print('pseudo_labels shape:', pseudo_labels.shape)
    # print('classwise_thresholds shape:', classwise_thresholds.shape)
    high_confidence_mask = probabilities.gather(1, pseudo_labels.unsqueeze(1)).squeeze(1) > classwise_thresholds[
        pseudo_labels]

    return high_confidence_mask


def compute_class_weights(pseudo_labels, num_classes):

    class_counts = torch.bincount(pseudo_labels, minlength=num_classes)
    class_counts = class_counts.float() + 1e-6
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()

    return class_weights


def weighted_cross_entropy_loss(probabilities, pseudo_labels, num_classes):


    class_weights = compute_class_weights(pseudo_labels, num_classes)
    ce_loss = F.cross_entropy(probabilities, pseudo_labels, reduction='none')
    weighted_loss = ce_loss * class_weights[pseudo_labels]

    return weighted_loss.mean()


def contrast_left_out(max_probs, device):
    """contrast_left_out

    If contrast_left_out, will select positive pairs based on
        max_probs > contrast_with_thresh, others will set to 0
        later max_probs will be used to re-weight the contrastive loss

    Args:
        max_probs (torch Tensor): prediction probabilities

    Returns:
        select_matrix: select_matrix with probs < contrast_with_thresh set
            to 0
    """
    contrast_mask = max_probs.ge(0.9).float()
    contrast_mask2 = torch.clone(contrast_mask)
    contrast_mask2[contrast_mask == 0] = -1
    select_elements = torch.eq(contrast_mask2.reshape([-1, 1]),
                               contrast_mask.reshape([-1, 1]).T).float()
    select_elements += torch.eye(contrast_mask.shape[0]).to(device)
    select_elements[select_elements > 1] = 1
    select_matrix = torch.ones(contrast_mask.shape[0]).to(device) * select_elements
    return select_matrix





if __name__ == "__main__":
    a = np.array([0, 0, 0, 1, 0, 1, 1, 0, 0,2])
    b = np.array([0, 1, 1, 1, 0, 0, 1, 0, 0,0])

    inter = AverageMeter()

    print(np.where(a == b))
    print(np.where(a == b)[0])
    hist1, edge1 = np.histogram(b, bins=np.arange(4))
    hist2, edge2 = np.histogram(a, bins=np.arange(4))

    print('hist1: ',hist1)
    print('hist2: ',hist2)

    inter.update(hist2)
    inter.update(hist1)

    print(inter.sum)
    print(np.unique(a))










