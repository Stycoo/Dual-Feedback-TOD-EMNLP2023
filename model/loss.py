import torch
import torch.nn.functional as F

def softCrossEntropy(inputs, target, reduction='mean'):
    log_likelihood = -F.log_softmax(inputs, dim=1)
    batch = inputs.shape[0]
    if reduction == 'mean':
        loss = torch.sum(torch.mul(log_likelihood, target)) / batch
    else:
        loss = torch.sum(torch.mul(log_likelihood, target))
    return loss

def kldivloss(score, gold_score):
    gold_score = torch.softmax(gold_score, dim=-1)
    score = F.log_softmax(score, dim=-1)
    loss_fct = torch.nn.KLDivLoss()
    return loss_fct(score, gold_score)

def masked_bce_cross_entropy(logits, labels, mask):
    node_num = mask.sum(-1)
    bce = torch.nn.BCELoss(reduction='none')
    bce_loss = bce(logits, labels)  # bsz, n_node
    bce_loss = torch.mean((bce_loss * mask).sum(-1)) / node_num  # bsz
    return bce_loss.mean()

def masked_mse_loss(logits, labels, mask):
    node_num = mask.sum(-1)
    mse_loss = torch.nn.MSELoss(reduction='none')
    loss = mse_loss(logits, labels)
    loss = torch.mean((loss * mask).sum(-1)) / node_num
    return loss.mean()

def info_nce_loss(features, bs, n_views, device, temperature=0.07):
    """ from https://github.com/sthalles/SimCLR.git
        features = (n_views*bs, dim)
        n_views: rgcn, transformer's view on user_rep
    """
    labels = torch.cat([torch.arange(bs) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(device)

    features = F.normalize(features, dim=1)
    similarity_matrix = torch.matmul(features, features.T)

    assert similarity_matrix.shape == (n_views * bs, n_views * bs)
    assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device)
    labels = labels[~mask].view(labels.shape[0], -1)

    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    assert similarity_matrix.shape == labels.shape
    # print(similarity_matrix)
    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device)
    return logits, labels
