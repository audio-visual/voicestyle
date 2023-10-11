
import torch
import cmpc2
import numpy as np 
import torch.nn as nn
from torch.nn import functional as F
def cosin_metric(x1, x2):
        #return np.dot(x1, x2) / (np.linalg.norm(x1) * np.linalg.norm(x2))
        return torch.sum(x1 * x2, dim=1) / (torch.norm(x1, dim=1) * torch.norm(x2, dim=1))

class CMPCLoss(torch.nn.Module):

    def __init__(self, opts):
        super(CMPCLoss, self).__init__()
        self.model = cmpc2.load(opts.cmpc2ckpt).to(opts.device)
        self.model.eval()
        self.logit_scale = nn.Parameter(torch.ones([]) * (1 / 0.03))
        

    def forward(self, audio, face):
        # 
        audio_emb,img_emb = self.model(audio,face)
        
        distance = 1-cosin_metric(audio_emb,img_emb)
        return distance
        
    
if __name__ == '__main__':
    import argparse
    args = argparse.Namespace()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    args.cmpc2ckpt = '/home/cwy/文档/backup/projects/CMPC/checkpoints/origin256/cmpc/model_best.pth.tar'
    cmpc_loss = CMPCLoss(args)

    img_size = 256

    visual_preprocess, audio_preprocess = cmpc2.visual_preprocess(img_size), cmpc2.audio_preprocess()

    # id10003/5ablueV_1tw/00013.wav
    # id10003/BQxxhq4539A/00017.wav
    wav_path = '/media/cwy/37620A9A581B65F3/data/VoxCeleb1/wav/id10008/CXgomMquVt8/00003.wav'
    frame_path = '/media/cwy/37620A9A581B65F3/data/VoxCeleb1/unzippedFaces/Aamir_Khan/1.6/5ablueV_1tw/0000800.jpg'

    audio = audio_preprocess(wav_path).unsqueeze(0).to(device) #(1,1,64,800)
    visual = visual_preprocess(frame_path).unsqueeze(0).to(device) #(1,3,224,224)

    distance = cmpc_loss(audio, visual)
    print(distance)


