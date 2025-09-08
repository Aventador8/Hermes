
from sklearn.metrics import accuracy_score

def cal_accuracy(prediction, label):
    prediction = prediction.argmax(dim=1)
    gt = label.cpu().detach().numpy()
    pred = prediction.cpu().detach().numpy()
    acc = accuracy_score(gt,pred)
    return acc

def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def accuracy_top1_accumulated(output, target, correct_count, total_samples):

    batch_size = target.size(0)

    _, pred = output.topk(1, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    correct_count += correct[:1].reshape(-1).float().sum(0).item()

    total_samples += batch_size

    return correct_count, total_samples

# y_true = np.array(['cat', 'dog', 'pig', 'cat', 'dog', 'pig'])
# y_pred = np.array(['cat', 'pig', 'dog', 'cat', 'cat', 'dog'])
#
#
# acc = accuracy_score(y_true, y_pred)
#
# precision, recall, f1 = precision_recall_fscore_support(y_true,y_pred,average='macro')[:-1]



