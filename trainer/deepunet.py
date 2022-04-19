from trainer.basetrainer import BaseTrainer
import os, time

class DeepUNetTrainer(BaseTrainer): 
    def __init__(self, mode, data_loader) -> None:
        super(DeepUNetTrainer, self).__init__(mode, data_loader)
        
        if mode == 'train':
            ctime = time.ctime().split()
            log_path = './log'
            if not os.path.exists(log_path):
                os.mkdir(log_path)
            
            log_dir = os.path.join(log_path, "%s_%s_%s_%s" % (ctime[-1], ctime[1], ctime[2], ctime[3]))
            os.mkdir(log_dir)
            self.log_file = open(os.path.join(log_dir, 'loss.txt'), 'w')
        
        self.save_path = './data/result'
        if not os.path.exists(self.save_path):
            os.mkdir(self.save_path)
        
        # set optimizer 
        self.optimizers = self.set_optimizer()