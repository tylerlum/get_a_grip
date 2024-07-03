import torch.nn as nn


class FCResBlock(nn.Module):
    """The FFHNet ResBlock.

    See: github.com/qianbot/FFHNet/blob/main/FFHNet/models/networks.py#L78
    """

    def __init__(self, Fin, Fout, n_neurons=256, use_bn: bool = True):
        super().__init__()
        self.Fin = Fin
        self.Fout = Fout

        self.fc1 = nn.Linear(Fin, n_neurons)
        self.bn1 = nn.BatchNorm1d(n_neurons)

        self.fc2 = nn.Linear(n_neurons, Fout)
        self.bn2 = nn.BatchNorm1d(Fout)

        if Fin != Fout:
            self.fc3 = nn.Linear(Fin, Fout)

        self.ll = nn.LeakyReLU(negative_slope=0.2)
        self.use_bn = use_bn

    def forward(self, x, final_nl=True):
        Xin = x if self.Fin == self.Fout else self.ll(self.fc3(x))

        Xout = self.fc1(x)
        if self.use_bn:
            Xout = self.bn1(Xout)
        Xout = self.ll(Xout)

        Xout = self.fc2(Xout)
        if self.use_bn:
            Xout = self.bn2(Xout)
        Xout = Xin + Xout

        if final_nl:
            return self.ll(Xout)
        return Xout
