#    Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from torch import nn


class MultipleOutputLoss2(nn.Module):
    def __init__(self, loss, weight_factors=None):
        """
        use this if you have several outputs and ground truth (both list of same len) and the loss should be computed
        between them (x[0] and y[0], x[1] and y[1] etc)
        :param loss:
        :param weight_factors:
        """
        super(MultipleOutputLoss2, self).__init__()
        self.weight_factors = weight_factors
        self.loss = loss

    def forward(self, x, y):
        assert isinstance(x, (tuple, list)), "x must be either tuple or list"
        assert isinstance(y, (tuple, list)), "y must be either tuple or list"
        if self.weight_factors is None:
            weights = [1] * len(x)
        else:
            weights = self.weight_factors

        l = weights[0] * self.loss(x[0], y[0])
        for i in range(1, len(x)):
            if weights[i] != 0:
                # l += weights[i] * self.loss(x[i], y[i])
                l += weights[i] * self.loss(x[i], y[i + 1])  ## nnUnet每次都是下采样2x， nnFormer首次下采样4x，decoder最后也是上采样4x
        return l
    
    # (Pdb) x[0].shape
    # torch.Size([2, 7, 96, 160, 160])
    # (Pdb) x[1].shape
    # torch.Size([2, 7, 24, 40, 40])
    # (Pdb) x[2].shape
    # torch.Size([2, 7, 12, 20, 20])
    # (Pdb) x[3].shape
    # torch.Size([2, 7, 6, 10, 10])
    # (Pdb) y[0].shape
    # torch.Size([2, 1, 96, 160, 160])
    # (Pdb) y[1].shape
    # torch.Size([2, 1, 48, 80, 80])
    # (Pdb) y[2].shape
    # torch.Size([2, 1, 24, 40, 40])
    # (Pdb) y[3].shape
    # torch.Size([2, 1, 12, 20, 20])
    # (Pdb) y[4].shape
    # torch.Size([2, 1, 6, 10, 10])
