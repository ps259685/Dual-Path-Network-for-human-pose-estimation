import torch
import torch.nn as nn
import numpy as np
import math


class JointsL2Loss(nn.Module):
    def __init__(self, has_ohkm=False, topk=8, thresh1=1, thresh2=0):
        super(JointsL2Loss, self).__init__()
        self.has_ohkm = has_ohkm
        self.topk = topk
        self.t1 = thresh1
        self.t2 = thresh2
        method = 'none' if self.has_ohkm else 'mean'
        self.calculate = nn.MSELoss(reduction=method)


    def forward(self, output, valid, label):
        assert output.shape == label.shape #output, label=[1, 17, 12, 12]
        batch_size = output.size(0)
        keypoint_num = output.size(1)
        loss = 0


        for i in range(batch_size):
            pred = output[i].reshape(keypoint_num, -1)
            gt = label[i].reshape(keypoint_num, -1)
            if not self.has_ohkm:
                weight = torch.gt(valid[i], self.t1).float()
                gt = gt * weight
            tmp_loss = self.calculate(pred, gt)

            #distribute
            if self.has_ohkm:
                tmp_loss = tmp_loss.mean(dim=1)
                weight = torch.gt(valid[i].squeeze(), self.t2).float()
                tmp_loss = tmp_loss * weight
                topk_val, topk_id = torch.topk(tmp_loss, k=self.topk, dim=0,
                        sorted=False)
                sample_loss = topk_val.mean(dim=0)
            else:
                sample_loss = tmp_loss

            loss = loss + sample_loss

        return loss / batch_size


if __name__ == '__main__':
    a = torch.randn(1, 17, 12, 12)
    b = torch.randn(1, 17, 12, 12)
    c = torch.randn(1, 17, 1) * 2
    weight = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    lasterrors = torch.tensor([2.4007, 2.0268, 1.9222, 1.8819, 2.1835, 2.2654, 2.0105, 1.6501, 1.8072,
        1.8634, 2.0230, 2.5501, 1.8857, 1.9303, 2.3238, 2.0897, 2.0688])
    loss = JointsL2Loss()
    #loss = JointsL2Loss(has_ohkm=True)
    device = torch.device('cuda')
    a = a.to(device)
    b = b.to(device)
    c = c.to(device)
    loss = loss.to(device)
    res = loss(a, c, b)
    #print(res)



