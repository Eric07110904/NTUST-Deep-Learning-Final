import glob, os, json
import torchvision 
from torch.utils.data.dataset import Dataset
from utils.color import make_colorgram_tensor
from utils.preprocess import scale
class AnimeDataset(Dataset):
    def __init__(self, path: str, transform) -> None:
        self.transform = transform
        self.filenames = glob.glob(path + "*.png")
        
    def __getitem__(self, index):
        user_id = self.filenames[index].split('\\')[-1][:-4]
        with open(os.path.join("./data/colorgram", "%s.json"% user_id), "r") as json_file:
            color_info = json.loads(json_file.read())
        colors = make_colorgram_tensor(color_info)
        img = torchvision.io.read_image(self.filenames[index])
        color_image = img[:,:,0:512]
        sketch = img[:,:,512:]
        if self.transform is not None: 
            color_image = self.transform(color_image)
            sketch = self.transform(sketch)
        return scale(sketch/255.0), scale(color_image/255.0), colors 
    
    def __len__(self):
        return len(self.filenames)

