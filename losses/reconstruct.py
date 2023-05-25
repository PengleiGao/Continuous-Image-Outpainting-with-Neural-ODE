import torch.nn as nn
class ReconLoss(nn.Module):
    def __init__(self,loss_type='l1'):
        super(ReconLoss, self).__init__()
        assert loss_type in ['l1','mse']
        if loss_type=='l1':
            self.loss=nn.L1Loss()
        else:
            self.loss=nn.MSELoss()

    def forward(self,input_fake, input_real):
        return self.loss(input_fake,input_real)

