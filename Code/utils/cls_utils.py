import torch
import torch.nn.functional as F

def get_classwise_thresholds(pseudo_labels, num_classes, base_threshold=0.95, min_threshold=0.5):
    class_counts = torch.bincount(pseudo_labels, minlength=num_classes)
    total_count = class_counts.sum().item()
    class_ratios = class_counts.float() / total_count
    classwise_thresholds = base_threshold - (base_threshold - min_threshold) * (1 - class_ratios)
    return classwise_thresholds


def filter_pseudo_labels_by_threshold(probabilities, pseudo_labels, classwise_thresholds):
    high_confidence_mask = probabilities.gather(1, pseudo_labels.unsqueeze(1)).squeeze(1) > classwise_thresholds[pseudo_labels]
    return high_confidence_mask


def compute_class_weights(pseudo_labels, num_classes):
    class_counts = torch.bincount(pseudo_labels, minlength=num_classes).float() + 1e-6
    class_weights = 1.0 / class_counts
    class_weights = class_weights / class_weights.sum()
    return class_weights


def weighted_cross_entropy_loss(probabilities, pseudo_labels, class_weights):
    ce_loss = F.cross_entropy(probabilities, pseudo_labels, reduction='none')
    weighted_loss = ce_loss * class_weights[pseudo_labels]
    return weighted_loss.mean()


def semi_supervised_training_step(probabilities_strong, probabilities_weak, pseudo_labels,
                                  num_classes, base_threshold=0.95, min_threshold=0.5, device=None):

    classwise_thresholds = get_classwise_thresholds(pseudo_labels, num_classes, base_threshold, min_threshold)


    high_confidence_mask = filter_pseudo_labels_by_threshold(probabilities_weak, pseudo_labels, classwise_thresholds)
    # print('pseudo_labels: ', pseudo_labels)
    # print('high_confidence_mask: ', high_confidence_mask)


    filtered_pseudo_labels = pseudo_labels[high_confidence_mask]
    filtered_probabilities = probabilities_strong[high_confidence_mask]

    if len(filtered_pseudo_labels) == 0:
        # print('filtered_pseudo_labels: ', len(filtered_pseudo_labels))
        loss = torch.tensor(0.0).to(device)
        return loss, high_confidence_mask


    class_weights = compute_class_weights(filtered_pseudo_labels, num_classes)


    loss = weighted_cross_entropy_loss(filtered_probabilities, filtered_pseudo_labels, class_weights)

    return loss, high_confidence_mask
