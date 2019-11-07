
def test_F_mse():
    import torch
    import torch.nn.functional as F
    import ubelt as ub
    ti = ub.Timerit(100, bestof=10, verbose=2)

    for timer in ti.reset('F.mse'):
        pred = torch.rand(1, 3, 16, 16).to(0)
        true = torch.rand(1, 3, 16, 16).to(0)
        weight = torch.rand(1, 3, 16, 16).to(0)
        torch.cuda.synchronize()
        with timer:
            result = F.mse_loss(pred, true, reduction='none')
            result = (result * weight).sum() / weight.sum()
            torch.cuda.synchronize()

    for timer in ti.reset('manual'):
        pred = torch.rand(1, 3, 16, 16).to(0)
        true = torch.rand(1, 3, 16, 16).to(0)
        weight = torch.rand(1, 3, 16, 16).to(0)
        torch.cuda.synchronize()
        with timer:
            result = (pred - true) ** 2
            result = (result * weight).sum() / weight.sum()
            torch.cuda.synchronize()
