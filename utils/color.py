import torch 
import numpy as np 
from torchvision import transforms 
from PIL import Image 
import colorgram 

def get_rgb(colorgram_result):
    """
    from colorgram_result, result rgb value as tuple of (r,g,b)
    """
    color = colorgram_result.rgb
    return (color.r, color.g, color.b)

def crop_region(image):
    """
    from image, crop 4 region and return 
    將一張圖片切成四等分 
    """
    width, height = image.size 
    h1 = height // 4 
    h2 = h1 + h1 
    h3 = h2 + h1 
    h4 = h3 + h1 
    image1 = image.crop((0, 0, width, h1))
    image2 = image.crop((0, h1, width, h2))
    image3 = image.crop((0, h2, width, h3))
    image4 = image.crop((0, h3, width, h4))
    return (image1, image2, image3, image4)

def img2colorinfo(img_path: str):
    img = Image.open(img_path).convert("RGB")
    img = transforms.Resize((512, 512))(img)
    images = list(crop_region(img))
    color_info = {}
    for i, img in enumerate(images, 1):
        color = colorgram.extract(img, 5)
        color_info[str(i)] = {
            "%d"%j: get_rgb(color[j]) for j in range(1, 5)
        }
    return color_info 

def make_colorgram_tensor(color_info, width=512, height=512):
    colors = list(color_info.values())
    color_num = len(colors[0].keys()) #抽取的顏色數量 
    tensor = np.ones([color_num * 3, height, width], dtype=np.float32) # 3 because every colorgram is RGB 
    region = height // 4 # 一張圖切成4個區塊
    
    for i, color in enumerate(colors): 
        index = region * i
        for j in range(1, color_num+1):
            """
                (i, j ) 整張圖片的第i個區段，的第j個顏色
            """
            r, g, b = color[str(j)]
            
            # assign channel index 
            red = (j - 1) * 3 
            green = (j - 1) * 3 + 1 
            blue = (j - 1) * 3 + 2 
            
            # assign value 
            tensor[red, index:index + region] *= r
            tensor[green, index:index + region] *= g
            tensor[blue, index:index + region] *= b 
    tensor = torch.from_numpy(tensor.copy())
    return tensor 