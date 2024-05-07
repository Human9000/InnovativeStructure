
class V2Flow(nn.Module):
    def __init__(self):
        super(V2Flow, self).__init__()
        self.raft_large = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False)
        weights = Raft_Large_Weights.DEFAULT
        self.transforms = weights.transforms()

    def preprocess(self, img1_batch, img2_batch):
        img1_batch = F.resize(img1_batch, size=[128, 128], antialias=False)
        img2_batch = F.resize(img2_batch, size=[128, 128], antialias=False)
        return self.transforms(img1_batch, img2_batch)

    @torch.no_grad()
    def forward(self, x): # c f w h
        self.eval()
        x1 = x.transpose(1, 0)
        img1_batch, img2_batch = x1[:-1], x1[1:]
        img1_batch, img2_batch = self.preprocess(img1_batch, img2_batch)
        list_of_flows = self.raft_large(img1_batch, img2_batch)
        predicted_flows = list_of_flows[-1]
        flow_imgs = flow_to_image(predicted_flows).transpose(0, 1)
        return flow_imgs.float()/255


class C3D_withTWO(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, num_classes, pretrained=False):
        super(C3D_withTWO, self).__init__()

        self.rgb = C3D(num_classes)
        self.opticalFlow_branch = C3D(num_classes)

        self.v2flow = V2Flow()
        self.softmax = nn.Softmax(dim=1)

    @torch.no_grad()
    def get_flows(self, x):
        # 映射回255的uint8图
        min = x.min()
        max = x.max()
        x2 = x - min
        x2 = x2 / (max - min)
        x2 = x2 * 255

        # 计算光流
        batch_flows = []
        for i in x2.to(torch.float32):  # c f w h
            flow_img = self.v2flow(i)
            batch_flows.append(flow_img)

        flows = torch.stack(batch_flows, dim=0)

        # 插值到输入的尺寸
        flows = nnF.interpolate(flows, x.shape[2:], mode='trilinear')
        return flows

    def forward(self, x):
        # rgb分支
        rgb_out = self.rgb(x)

        # 光流分支
        opticalFlow_out = self.opticalFlow_branch(self.get_flows(x))

        # 自注意力融合
        softmax1 = self.softmax(rgb_out*opticalFlow_out)
        multiplication1 = softmax1*rgb_out
        final_out = self.softmax(multiplication1 + opticalFlow_out)
        return final_out


