from numpy import dtype
import torch.nn as nn
import torch
import torch.nn.functional as F


class NeuralNet(nn.Module):

    def __init__(self, layer_sizes):
        super(NeuralNet, self).__init__()
        layers = []
        input_features = 7
        for i, output_features in enumerate(layer_sizes):
            layers.append(nn.Linear(input_features, output_features))
            #layers.append(nn.BatchNorm1d(output_features))
            layers.append(nn.ReLU())
            input_features = output_features
        self.layers = nn.Sequential(*layers)

    def forward(self, forces, coords, index_list):  # [B,3] [B,P,3], [B,P]
        B = forces.shape[0]
        P = coords.shape[1]

        


        index_list = index_list.view(B,P,1)                     # [B,P,1]
        forces = forces.view(B,1,3).repeat(1,P,1)               # [B,P,3]
        a = torch.cat((forces,coords, index_list), dim=2)       # [B,P,7]

        

        out = self.layers(a)                        # [B,P,4]
        disp_pred = out[:,:,:3]                     # [B,P,3]
        stress_pred = out[:,:,3]                    # [B,P]

        return disp_pred, stress_pred
