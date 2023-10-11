
ref_folder = '/media/cwy/37620A9A581B65F3/data/VoxCeleb1/unzippedFaces/Adrianne_Palicki/1.6/6RiWT7JoAkk/*.jpg'
folder = '/home/cwy/文档/data/VGG_ALL_FRONTAL/Adrianne_Palicki/*.jpg'

import torch
import clip
from PIL import Image
import glob

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)
# TODO combine
ref_image_feature_list = []
with torch.no_grad():
    for path in glob.glob(ref_folder):
        ref_image = preprocess(Image.open(path)).unsqueeze(0).to(device)


        ref_image_features = model.encode_image(ref_image)
        ref_image_feature_list.append(ref_image_features)
    # text_features = model.encode_text(text)
ref_features = torch.stack(ref_image_feature_list).mean(dim=0)
# print(ref_features.shape)

simi_list = []
def argsort(seq):
    return [x for x,y in sorted(enumerate(seq), key = lambda x: x[1])]

with torch.no_grad():
    for image_path in glob.glob(folder):
        image = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        simi = ref_features@image_features.t()
        simi_list.append(simi.item())
# print()
select_paths = glob.glob(folder)[argsort(simi_list)[-1]]
print(select_paths)



    
        