class BaseTrainer:
    def __init__(self, mode, data_loader) -> None:
        """
        Constructor
        """
        self.mode = mode  
        self.data_loader = data_loader 
        self.device = "cuda"
        self.resolution = 512 
    
    def train(self):
        raise NotImplementedError
    
    def validate(self, dataset, epoch, samples=3):
        raise NotImplementedError 

    def test(self):
        raise NotImplementedError 
    
    def save_model(self, name, epoch):
        raise NotImplementedError 

    def set_optimizer(self):
        raise NotImplementedError
    
    def set_losses(self):
        raise NotImplementedError
    
    def update_G(self):
        raise NotImplementedError 
    
    def update_D(self):
        raise NotImplementedError