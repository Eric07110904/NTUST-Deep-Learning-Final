import glob 
import torchvision 
from torch.utils.data.dataset import Dataset

class AnimeDataset(Dataset):
    def __init__(self, path: str, transform) -> None:
        self.transform = transform
        self.filenames = glob.glob(path + "*.png")
        
    def __getitem__(self, index):
        user_id = self.filenames[index].split('\\')[-1][:-4]
        img = torchvision.io.read_image(self.filenames[index])
        color_image = img[:,:,0:512]
        sketch = img[:,:,512:]
        if self.transform is not None: 
            color_image = self.transform(color_image)
            sketch = self.transform(sketch)
        return sketch, color_image
    
    def __len__(self):
        return len(self.filenames)

