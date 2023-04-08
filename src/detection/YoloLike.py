import time

import torch

def get_det_output(max_numobjs, num_classes,pos_bits):
    return max_numobjs * (num_classes + (4*pos_bits))



def softmaximum(x1, x2):
    m = torch.maximum(x1, x2)
    sm = m+ ((x1- m).exp() + (x2-m).exp()).log()
    return sm
class Fold(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self,x):
        h1 = torch.relu(x)
        h2 = torch.relu(-x)
        y = torch.cat([h1,h2],dim=1)
        return y


class EnergyDetect(torch.nn.Module):
    def __init__(self, num_objects, num_classes, num_layers, img_shape, out_channel=64):
        super().__init__()
        self.num_layers = num_layers
        self.img_shape = img_shape
        bits, spots = 10, 4
        self.out_shape = num_objects * (num_classes + spots * bits)
        self.hidden_outchannel = out_channel
        self.forward_layers = []
        self.shortcut_layers = []
        self.nl = Fold()
        self.build()

    def forward(self,x):
        y = None
        for forward_layer,shortcut_layer in list(zip(self.forward_layers, self.shortcut_layers)):
            h = forward_layer(x)
            x = self.nl(h)
            h1 = shortcut_layer(x)
            h1= h1.logsumexp(dim=(2,3),keepdim=True)
            if y is None:
                y = h1
            else:
                y = softmaximum(h1, y)
        y = y.squeeze(dim=(2,3))
        return y

    def build(self):
        in_channels = self.img_shape[1]

        for i_layer in range(self.num_layers):
            l = torch.nn.Conv2d(in_channels,self.outchannel,3)
            s = torch.nn.Conv2d(in_channels,self.out_shape,3)
            self.forward_layers.append(l)
            self.shortcut_layers.append(s)

        return

if __name__ == '__main__':
    import cv2
    vc = cv2.VideoCapture(0)
    model = None
    while True:
        ret, img = vc.read()
        if not ret:
            time.sleep(0.01)
        else:
            print(img)
            # cv2.imshow("A", img)
            cv2.waitKey(1)

