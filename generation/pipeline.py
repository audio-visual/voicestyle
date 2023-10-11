import math 
import argparse
import cmpc2 
import clip
import torch 
from torch import optim
import os 
from model import Generator
from tqdm import tqdm
from PIL import Image
from similarity.cmpc_loss import CMPCLoss
from projector import get_lr, latent_noise, noise_normalize_, noise_regularize, make_image,project2latent


# inference-time optimization
def optimize(latent_code_init, audio_inputs,args):
    g_ema = Generator(args.size, 512, 8)
    print('load checkpoint:', args.styleganckpt)
    g_ema.load_state_dict(torch.load(args.styleganckpt)["g_ema"], strict=False)
    g_ema.eval()
    g_ema = g_ema.to(args.device)  

    cmpc_loss = CMPCLoss(args)

    n_mean_latent = 10000
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=args.device)
        latent_out = g_ema.style(noise_sample)
        # print('latent out shape:', latent_out.shape)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5


    latent_in = latent_code_init.detach().clone()

    if args.w_plus:
        latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(1, 1, 1, 1).normal_())
    
    latent_in.requires_grad = True
    # print('latent in shape:', latent_in.shape)
    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=args.optimize_lr)

    pbar = tqdm(range(args.step))
    latent_path = []

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.optimize_lr)
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

        c_loss = cmpc_loss(audio_inputs, img_gen.clamp_(min=-1, max=1))
        n_loss = noise_regularize(noises)
        r_loss = ((latent_in - latent_n) ** 2).sum()
        # print(r_loss)

        loss =  c_loss + args.noise_regularize * n_loss  + args.latent_regularize * r_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        if (i + 1) % 20 == 0:
            latent_path.append(latent_in.detach().clone())
        
        pbar.set_description(
            (
                f"similarity: {c_loss.item():.4f}; noise regularize: {n_loss.item():.4f}; latent regularize: {args.latent_regularize * r_loss.item():.4f}"
            )
        )

    pil_imgs = []    
        
    # print(latent_path[-1].shape)
    for i in range(len(latent_path)):
        img_gen, _ = g_ema([latent_path[i]], input_is_latent=True, noise=noises)

        # filename = os.path.splitext(os.path.basename(files[0]))[0] + ".pt"
        # print(img_gen.shape)

        img_ar = make_image(img_gen)

        pil_img = Image.fromarray(img_ar[0])
        pil_imgs.append(pil_img)
        # pil_img.save('generated_img_{}.png'.format(i))

    return pil_imgs
        
def argsort(seq):
    return [x for x,y in sorted(enumerate(seq), key = lambda x: x[1])]

def replace_path(k_nearest_image_paths, args):
    clip_model, preprocess = clip.load("ViT-B/32", device=args.device)
    final_result = []
    for unfrontal_face_path in k_nearest_image_paths:
        ref_paths = os.path.join('../voxceleb/unzippedFaces/',unfrontal_face_path,'*.jpg')
        ref_path = glob.glob(ref_paths)[0]
        # print(ref_path)
        ref_image = preprocess(Image.open(ref_path)).unsqueeze(0).to(device)
        with torch.no_grad():
            ref_image_features = clip_model.encode_image(ref_image)

        tmp = os.path.join('../voxceleb/VGG_ALL_FRONTAL', unfrontal_face_path.split(os.path.sep)[0],'*.jpg')
        imgs = glob.glob(tmp)
        simi_list = []
        with torch.no_grad():
            for image_path in imgs:
                image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
                image_features = clip_model.encode_image(image)
                simi = ref_image_features@image_features.t()
                simi_list.append(simi.item())
        final_result.append(imgs[argsort(simi_list)[-1]])
    return final_result

if __name__ == '__main__':
    import glob
    args = argparse.Namespace()

    args.w_plus=False
    args.step=100 # optimize_step
    args.device= "cuda" if torch.cuda.is_available() else "cpu"
    args.size= 256
    args.lr = 0.1
    args.optimize_lr = 0.01
    args.noise = 0.05
    args.noise_ramp = 0.75
    args.noise_regularize = 1e5
    args.latent_regularize = 0.001
    args.cmpc2ckpt = '../weights/model_best_256.pth.tar'
    memory_path = '../weights/frame_memory_best_256.pth.tar'
    df_path = '../weights/training.csv'
    args.styleganckpt = '../weights/550000.pt'

    
    audio_preprocess = cmpc2.audio_preprocess()
    model = cmpc2.load(args.cmpc2ckpt).to(args.device)
    # you should manually change the model mode
    model.eval()
    frame = torch.randn(1,3,256,256).to(args.device)

    # wav_path = '/media/cwy/37620A9A581B65F3/data/VoxCeleb1/wav/id10008/CXgomMquVt8/00003.wav'
    # wav_path = '/media/cwy/37620A9A581B65F3/data/VoxCeleb1/wav/id10014/6RiWT7JoAkk/00003.wav'
    # wav_path = '/media/cwy/37620A9A581B65F3/data/VoxCeleb1/wav/id11251/5-6lI5JQtb8/00001.wav'
    # wav_path = '/media/cwy/37620A9A581B65F3/data/VoxCeleb1/wav/id10153/81LUckMu7qI/00003.wav'
    # wav_path = '/media/cwy/37620A9A581B65F3/data/VoxCeleb1/wav/id10153/LgWFAJ6C98Q/00001.wav'
    wav_path = '../voxceleb/wav/id10845/0RsOMTn-DSM/00001.wav'
    # wav_path = '/media/cwy/37620A9A581B65F3/data/VoxCeleb1/wav/id10883/H4nXd3zgK1Y/00001.wav'
    # wav_path = '/media/cwy/37620A9A581B65F3/data/VoxCeleb1/wav/id10724/DP5gCFoWgGI/00001.wav'

    fbank = audio_preprocess(wav_path).unsqueeze(0).to(args.device).detach()

   
    with torch.no_grad():
        audio_emb, _ = model(fbank,frame)

        
    k_nearest_image_paths = cmpc2.getKNearestSamples(audio_emb,memory_path,df_path,cluster=1000,K=3)[:8] # you can change 8 to other number, 10 or 15,...
    # print('finded image length:',len(k_nearest_image_paths))
    

    # replace image paths to frontal face 
    final_result = replace_path(k_nearest_image_paths, args)
    print('replaced paths:')
    print(final_result)
    
    
    init_latents = project2latent(ckpt=args.styleganckpt, files=final_result,size=args.size, step=300)
    # # you can apply other aggregation method
    combine_latent = torch.mean(init_latents, dim=0,keepdim=True)
  
    
    pil_images = optimize(latent_code_init=combine_latent, audio_inputs=fbank, args=args) 
    pil_images[-1].save('../results/{}_{}.png'.format(wav_path.split(os.path.sep)[-3],wav_path.split(os.path.sep)[-2]))

