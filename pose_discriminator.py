


import torch.nn as nn

class Pos2dDiscriminator(nn.Module):
    def __init__(self, num_joints=16):
        super(Pos2dDiscriminator, self).__init__()

        # Pose path
        self.pose_layer_1 = nn.Linear(num_joints * 2, 100)
        self.pose_layer_2 = nn.Linear(100, 100)
        self.pose_layer_3 = nn.Linear(100, 100)
        self.pose_layer_4 = nn.Linear(100, 100)

        self.layer_last = nn.Linear(100, 100)
        self.layer_pred = nn.Linear(100, 1)

        self.relu = nn.LeakyReLU()

    def init_weights_real(m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight)

    def init_weights(self):
        self.apply(self.init_weights_real)

    def forward(self, x):
        # Pose path
        x = x.contiguous().view(x.size(0), -1)
        d1 = self.relu(self.pose_layer_1(x))
        d2 = self.relu(self.pose_layer_2(d1))
        d3 = self.relu(self.pose_layer_3(d2) + d1)
        d4 = self.pose_layer_4(d3)

        d_last = self.relu(self.layer_last(d4))
        d_out = self.layer_pred(d_last)
        return d_out