# -*- coding: utf-8 -*-
import torch  # NOQA
import torch.nn.functional as F
import torch.nn.modules
import numpy as np
from torch import autograd
from packaging import version


_HAS_REDUCTION = version.parse(torch.__version__) >= version.parse('0.4.1')


def one_hot_embedding(labels, num_classes):
    """
    Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N,#classes].

    References:
        https://discuss.pytorch.org/t/convert-int-into-one-hot-format/507/4

    CommandLine:
        python -m netharn.loss one_hot_embedding

    Example:
        >>> # each element in target has to have 0 <= value < C
        >>> labels = torch.LongTensor([0, 0, 1, 4, 2, 3])
        >>> num_classes = max(labels) + 1
        >>> t = one_hot_embedding(labels, num_classes)
        >>> assert all(row[y] == 1 for row, y in zip(t.numpy(), labels.numpy()))
        >>> import ubelt as ub
        >>> print(ub.repr2(t.numpy().tolist()))
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
        ]
        >>> t2 = one_hot_embedding(labels.numpy(), num_classes)
        >>> assert np.all(t2 == t.numpy())
        >>> if torch.cuda.is_available():
        >>>     t3 = one_hot_embedding(labels.to(0), num_classes)
        >>>     assert np.all(t3.cpu().numpy() == t.numpy())
    """
    # if True:
    if torch.is_tensor(labels):
        y = torch.eye(int(num_classes), device=labels.device)
        y_onehot = y[labels]
    else:
        y = np.eye(int(num_classes))
        y_onehot = y[labels]
    return y_onehot
    # else:
    #     # y = torch.eye(num_classes)  # [D,D]
    #     # if labels.is_cuda:
    #     #     y = y.cuda(labels.get_device())
    #     # y_onehot = y[labels]        # [N,D]
    #     # if cpu:
    #     y = torch.eye(int(num_classes))  # [D,D]
    #     y_onehot = y[labels.cpu()]  # [N,D]
    #     if labels.is_cuda:
    #         device = labels.get_device()
    #         y_onehot = y_onehot.cuda(device)
    #     # else:
    #     #     if labels.is_cuda:
    #     #         y_onehot = torch.cuda.FloatTensor(labels.shape[0], num_classes,
    #     #                                           device=labels.get_device()).zero_()
    #     #         y_onehot.scatter_(1, labels[:, None], 1)
    #     #     else:
    #     #         y_onehot = torch.FloatTensor(labels.shape[0], num_classes).zero_()
    #     #         y_onehot.scatter_(1, labels[:, None], 1)
    #     return y_onehot


def _backwards_compat_reduction_kw(size_average, reduce, reduction):
    if size_average is None and reduce is None:
        if reduction == 'none':
            size_average = False
            reduce = False
        elif reduction == 'elementwise_mean':
            size_average = True
            reduce = True
        elif reduction == 'sum':
            size_average = False
            reduce = True
        else:
            raise ValueError(reduction + " is not a valid value for reduction")
    else:
        if size_average is None or reduce is None:
            raise Exception(
                'Must specify both size_average and reduce in '
                'torch < 0.4.1, or specify neither and use reduction')
        else:
            if not size_average and not reduce:
                reduction = 'none'
            elif size_average and not reduce:
                reduction = 'none'
            elif size_average and reduce:
                reduction = 'elementwise_mean'
            elif not size_average and reduce:
                reduction = 'sum'
            else:
                raise ValueError(
                    'Impossible combination of size_average and reduce')
    return size_average, reduce, reduction


class FocalLoss(torch.nn.modules.loss._WeightedLoss):
    r"""

    Original implementation in [1]

    # Math:
    #     FL(p[t]) = -α[t] * (1 − p[t]) ** γ * log(p[t]).

    .. math::
        FL(p_t) = - \alpha_t * (1 − p[t]) ** γ * log(p[t]).
        focal_loss(x, class) = weight[class] * (-x[class] + log(\sum_j exp(x[j])))

    Args:
        focus (float): Focusing parameter. Equivelant to Cross Entropy when
            `focus == 0`. (Defaults to 2) (Note: this is gamma in the paper)

        weight (Tensor, optional): a manual rescaling weight given to each
           class. If given, it has to be a Tensor of size `C`. Otherwise, it is
           treated as if having all ones.


           Finally we note that α, the weight assigned to the rare class, also
           has a stable range, but it interacts with γ making it necessary to
           select the two together

           This should be set depending on `focus`. See [2] for details.
           In general α should be decreased slightly as γ is increased
           (Note: this is α in the paper)

           α ∈ [0, 1] for class 1 and 1−α for class −1

        size_average (bool, optional): By default, the losses are averaged
           over observations for each minibatch. However, if the field
           size_average is set to ``False``, the losses are instead summed for
           each minibatch. Ignored when reduce is ``False``. Default: ``True``

        reduce (bool, optional): By default, the losses are averaged or summed
           for each minibatch. When reduce is ``False``, the loss function returns
           a loss per batch element instead and ignores size_average.
           Default: ``True``

        ignore_index (int, optional): Specifies a target value that is ignored
            and does not contribute to the input gradient. When size_average is
            ``True``, the loss is averaged over non-ignored targets.

    References:
        [1] https://github.com/kuangliu/pytorch-retinanet/blob/master/loss.py
        [2] https://arxiv.org/abs/1708.02002

    SeeAlso:
        https://github.com/marvis/pytorch-yolo2/blob/master/FocalLoss.py
        https://discuss.pytorch.org/t/how-to-implement-focal-loss-in-pytorch/6469/11

    Example:
        >>> self = FocalLoss(reduction='none')
        >>> # input is of size N x C
        >>> N, C = 8, 5
        >>> data = autograd.Variable(torch.randn(N, C), requires_grad=True)
        >>> # each element in target has to have 0 <= value < C
        >>> target = autograd.Variable((torch.rand(N) * C).long())
        >>> input = torch.nn.LogSoftmax(dim=1)(data)
        >>> #self.focal_loss_alt(input, target)
        >>> self.focal_loss(input, target)
        >>> output = self(input, target)
        >>> output.sum().backward()

        input = torch.FloatTensor([
            [0, 1, 0, 0],
            [0, .9, 0, 0],
            [0, .98, 0, 0],
            [.7, .21, .1, .1],
            [.3, .3, .3, .1],
            [0, 1, 0, 0],
            [0, .9, 0, 0],
            [0, .98, 0, 0],
            [.7, .21, .1, .1],
            [.3, .3, .3, .1],
        ]) * 10
        target = torch.LongTensor([1, 1, 1, 1, 1, 0, 2, 3, 3, 3])
        target = autograd.Variable(target)
        input = autograd.Variable(input)

        weight = torch.FloatTensor([1, 1, 1, 10])
        self = FocalLoss(reduce=False, weight=weight)
    """

    def __init__(self, focus=2, weight=None, size_average=None, reduce=None,
                 reduction='elementwise_mean', ignore_index=-100):

        size_average, reduce, reduction = _backwards_compat_reduction_kw(
            size_average, reduce, reduction)
        if _HAS_REDUCTION:
            super(FocalLoss, self).__init__(weight=weight,
                                            # reduce=reduce,
                                            # size_average=size_average,
                                            reduction=reduction)
        else:
            super(FocalLoss, self).__init__(weight=weight, reduce=reduce,
                                            size_average=size_average)
            self.size_average = size_average  # fix for travis?
            self.reduce = reduce

        self.focus = focus
        self.ignore_index = ignore_index

    def focal_loss(self, input, target):
        """
        Focal loss standard definition.

        Args:
          input: (tensor) sized [N,D].
          target: (tensor) sized [N,].

        Return:
          (tensor) sized [N,] focal loss for each class

        CommandLine:
            python -m netharn.loss FocalLoss.focal_loss:0 --profile
            python -m netharn.loss FocalLoss.focal_loss:1 --profile

        Example:
            >>> # input is of size N x C
            >>> import numpy as np
            >>> N, C = 8, 5
            >>> # each element in target has to have 0 <= value < C
            >>> target = autograd.Variable((torch.rand(N) * C).long())
            >>> input = autograd.Variable(torch.randn(N, C), requires_grad=True)
            >>> # Check to be sure that when gamma=0, FL becomes CE
            >>> loss0 = FocalLoss(reduction='none', focus=0).focal_loss(input, target)
            >>> #loss1 = F.cross_entropy(input, target, reduction='none')
            >>> loss1 = F.cross_entropy(input, target, size_average=False, reduce=False)
            >>> #loss2 = F.nll_loss(F.log_softmax(input, dim=1), target, reduction='none')
            >>> loss2 = F.nll_loss(F.log_softmax(input, dim=1), target, size_average=False, reduce=False)
            >>> assert np.all(np.abs((loss1 - loss0).data.numpy()) < 1e-6)
            >>> assert np.all(np.abs((loss2 - loss0).data.numpy()) < 1e-6)
            >>> lossF = FocalLoss(reduction='none', focus=2, ignore_index=0).focal_loss(input, target)
            >>> weight = torch.rand(C)
            >>> lossF = FocalLoss(reduction='none', focus=2, weight=weight, ignore_index=0).focal_loss(input, target)
        """
        if self.weight is None:
            alpha = 1
        else:
            # Create a per-input weight
            alpha = autograd.Variable(self.weight)[target]

        # Compute log(p) for NLL-loss.
        nll = F.log_softmax(input, dim=1)  # [N,C]

        gamma = self.focus

        num_classes = input.shape[1]

        # remove any loss associated with ignore_label
        mask = (target != self.ignore_index).float()  # [N,]

        # Determine which entry in nll corresponds to the target
        t = one_hot_embedding(target.data, num_classes)  # [N,C]
        t = autograd.Variable(t)

        # We only need the log(p) component corresponding to the target class
        target_nll = (nll * t).sum(dim=1)  # [N,]  # sameas nll[t > 0]
        target_p = torch.exp(target_nll)   # [N,]

        # Reduce the weight of easy examples
        hardness = (1 - target_p)               # [N,]
        w = alpha * hardness.pow(gamma)  # [N,]

        # Normal cross-entropy computation (but with augmented weights per example)
        output = -w * mask * target_nll  # [N,]
        return output

    def forward(self, input, target):
        """
        Args:
          input: (tensor) predicted class confidences, sized [batch_size, #classes].
          target: (tensor) encoded target labels, sized [batch_size].

        Returns:
            (tensor) loss
        """
        output = self.focal_loss(input, target)
        if _HAS_REDUCTION:
            if self.reduction == 'elementwise_mean':
                output = output.sum()
                output = output / input.shape[0]
            elif self.reduction == 'sum':
                output = output.sum()
        else:
            if self.reduce:
                output = output.sum()
                if self.size_average:
                    output = output / input.shape[0]
        return output


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m netharn.loss
    """
    import xdoctest
    xdoctest.doctest_module(__file__)
