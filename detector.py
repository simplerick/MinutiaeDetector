import torch
from utils import get_coord, compute_image_relative_coord


class YOLO(torch.nn.Module):
    """
    YOLO-like model class for detecting minutiae.
    """
    def __init__(self, size, inter_dist):
        super().__init__()
        self.size = size
        self.inter_dist = inter_dist
        self.total_stride = 2*2*3 #based on the model below
        self.net = torch.nn.Sequential(torch.nn.Conv2d(1,16,5,stride=2, bias=False),
                                    torch.nn.BatchNorm2d(16),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(16,32,5, bias=False),
                                    torch.nn.BatchNorm2d(32),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(32,64,3, bias=False),
                                    torch.nn.BatchNorm2d(64),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(64,128,3, stride=2, bias=False),
                                    torch.nn.BatchNorm2d(128),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(128,256,3, bias=False),
                                    torch.nn.BatchNorm2d(256),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(256,512,3, bias=False),
                                    torch.nn.BatchNorm2d(512),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(512,1024,3, bias=False),
                                    torch.nn.BatchNorm2d(1024),
                                    torch.nn.ReLU(),
                                    torch.nn.AvgPool2d(3),
                                    # torch.nn.ReLU(),
                                    # 1x1
                                    torch.nn.Conv2d(1024,800,1),
                                    torch.nn.ReLU(),
                                    torch.nn.Conv2d(800,3,1),
                                   )


    def forward(self,x):
        return self.net(x)


    def predict(self, image,threshold=0.5):
        h,w = image.shape[:2]
        device = next(self.parameters()).device
        img = torch.tensor(image, dtype=torch.float).reshape(1,1,h,w).to(device)
        # after applying model we get [1, 3, (h-size)//total_stride + 1, (w-size)//total_stride + 1] tensor
        out = self.forward(img).detach().squeeze().permute(2,1,0)
        # prob.shape = [number of patches]
        prob = torch.sigmoid(out[...,0]).cpu().numpy().reshape(-1)
        mask = prob >= threshold
        # patch relative coord
        coord = get_coord(out[...,1:].cpu().numpy(), self.size, self.inter_dist)
        # image relative coord, coord.shape = [number of patches, 2]
        coord = compute_image_relative_coord(coord, self.total_stride).reshape(-1,2)
        # apply mask
        return coord[mask], prob[mask]



def loss(logits, y_target, c=0.1):
    logits = logits.squeeze()
    l = c*torch.nn.functional.binary_cross_entropy_with_logits(logits[:,0], y_target[:,0])
    l += (y_target[:,0].unsqueeze(-1)*(logits[:,1:]-y_target[:,1:])**2).mean()
    return l
