import torch
from config import args
import data
import model
import loss
import utility
import numpy as np
from trainer import Trainer

#Set random seed

torch.manual_seed(args.seed)
np.random.seed(args.seed)
checkpoint = utility.checkpoint(args)

def main():
    global model

    if checkpoint.ok:
        loader = data.Data(args)
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only and not args.demo else None
        t = Trainer(args, loader, _model, _loss, checkpoint)
        while not t.terminate():
            t.train()
            t.test()
            
        checkpoint.done()

if __name__ == '__main__':
    main()

    '''
    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py
    '''