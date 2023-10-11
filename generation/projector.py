import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

import lpips
import facial_recognition
from model import Generator


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )

"""
python projector.py --ckpt [CHECKPOINT] --size [GENERATOR_OUTPUT_SIZE] FILE1 FILE2
"""

def project2latent(ckpt,files,size=256, step=500):

    args = argparse.Namespace()
    device = "cuda"
        
    # /home/stylegan2-ffhq-config-f.pt ,1024
    # /home/550000.pt, 256
    
    # lr_rampup = 0.05
    # lr_rampdown = 0.25
    args.lr = 0.1
    args.noise = 0.05
    args.noise_ramp = 0.75
    args.noise_regularize = 1e5
    args.mse = 0
    w_plus = False
    

    n_mean_latent = 10000

    # resize = min(args.size, 256) 
    resize = 256 # if the original image size is smaller than 256, I found lpips perform better on 256 size

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]), # 变成了[-1,1]之间
        ]
    )

    imgs = []

    for imgfile in files:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs.append(img)

    imgs = torch.stack(imgs, 0).to(device)

    g_ema = Generator(size, 512, 8)
    print('load checkpoint:', ckpt)
    g_ema.load_state_dict(torch.load(ckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(device)

    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)
        # print('latent out shape:', latent_out.shape)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )
    id_loss = facial_recognition.IDLoss(use_gpu=device.startswith("cuda"))

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

    latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)

    if w_plus:
        latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

    latent_in.requires_grad = True
    # print('latent in shape:', latent_in.shape)
    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    pbar = tqdm(range(step))
    latent_path = []

    for i in pbar:
        t = i / step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        # print('noise_strength shape:', noise_strength.shape)
        latent_n = latent_noise(latent_in, noise_strength.item())

        img_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

        batch, channel, height, width = img_gen.shape
        if i==0:
            print('stylegan generated image shape:',height,width)

        if height > 256:
            factor = height // 256

            img_gen = img_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            img_gen = img_gen.mean([3, 5])

        p_loss = percept(img_gen, imgs).sum()
        # print('ori img:',torch.max(imgs))
        # print('gen img:',torch.max(img_gen))
        identity_loss = id_loss(img_gen,imgs)[0]
        
        n_loss = noise_regularize(noises)
        mse_loss = F.mse_loss(img_gen, imgs)

        # the original loss: the result is 550000_align-000015-project_256_step500_perceptual_loss.png
        # loss =  p_loss + args.noise_regularize * n_loss + args.mse * mse_loss 

        # we found this loss can produce more fine-grained image: the result is 550000_align-000015-project_256_step500_id_perceptual_loss.png
        # however, if only use id loss, the generated image would change: the result is 550000_align-000015-project_256_step500_id_loss.png
        loss =  p_loss + args.noise_regularize * n_loss + args.mse * mse_loss + identity_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        if (i + 1) % 100 == 0:
            latent_path.append(latent_in.detach().clone())

        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
            )
        )
    # print(latent_path[-1].shape)
    img_gen, _ = g_ema([latent_path[-1]], input_is_latent=True, noise=noises)

    # filename = os.path.splitext(os.path.basename(files[0]))[0] + ".pt"

    # img_ar = make_image(img_gen)

    # result_file = {}
    # for i, input_name in enumerate(files):
    #     noise_single = []
    #     for noise in noises:
    #         noise_single.append(noise[i : i + 1])
        
    #     # print(noise_single[0].shape)
    #     # print(noise_single[1].shape)

    #     result_file[input_name] = {
    #         "img": img_gen[i],
    #         "latent": latent_in[i],
    #         "noise": noise_single,
    #     }

    #     img_name = os.path.splitext(os.path.basename(ckpt))[0]+ '_'+os.path.splitext(os.path.basename(input_name))[0] + "-project_{}_step{}.png".format(size, step)
    #     pil_img = Image.fromarray(img_ar[i])
        # pil_img.save(img_name)
    
    # combine_latent = torch.mean(latent_in, dim=0,keepdim=True)
    # print('combine_latent',combine_latent.shape)
    # combine_noise = torch.stack(noises).mean(dim=0)
    
    # combine_img_gen, _ = g_ema([combine_latent], input_is_latent=True, noise=None)
    # combine_img_ar = make_image(combine_img_gen)
    # pil_img = Image.fromarray(combine_img_ar[0])
    # pil_img.save('combine_nonoise.png')

    return latent_in

    # torch.save(result_file, filename)
    # TODO 存单个图像和pt，存均值图像和pt


if __name__ =='__main__':
    import glob
    
    folder = '/home/cwy/data/VGG_ALL_FRONTAL/Aaron_Staton'
    imgs = glob.glob(os.path.join(folder,'*.jpg'))
    print(imgs[:3])
    ckpt = '/media/cwy/sdb1/data/weights/550000.pt'
    project2latent(ckpt=ckpt, files=imgs[:3],step=500)

