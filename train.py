import argparse
import torch
from dataset.data import AudioDataLoader, AudioDataset
from src.trainer import Trainer
from model.sepformer import Sepformer
import numpy as np
from adamp import AdamP, SGDP


def main ( config ) :
    torch.manual_seed(0)                                       # Random Seed
    np.random.seed(0)                                          # Random Seed

    # data
    tr_dataset = AudioDataset(json_dir = "./json/tr",  # The directory contains mix.json, s1.json, s2.json
                              batch_size = 1,                  # Betch Size
                              sample_rate = 8000,              # Sampling Rate
                              segment = -1,                    # Voice duration in sec
                              cv_max_len = 10 )                # CV max Length

    cv_dataset = AudioDataset(json_dir = "./json/cv",  
                              batch_size = 1,
                              sample_rate = 8000,
                              segment = -1,
                              cv_max_len = 10)

    tr_loader = AudioDataLoader(tr_dataset,
                                batch_size = 1,
                                shuffle = True,
                                num_workers = 0
                                )

    cv_loader = AudioDataLoader(cv_dataset,
                                batch_size = 1,
                                shuffle = True,
                                num_workers = 0)

    data = {"tr_loader": tr_loader, 
            "cv_loader": cv_loader}

    # Model Initialyzation
    model = Sepformer(  N = 128,       # Embeding Dimension
                        C = 2,         # 
                        L = 16,        # Karnel Size
                        H = 8,         # Head size
                        K = 250,       # Chunk Size
                        Global_B = 1,  # Global Loop
                        Local_B = 2)   # Local loop

    print("{:.3f} million".format(sum([param.nelement() for param in model.parameters()]) / 1e6))

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model)
        model.cuda()

    optimizer_type = "adamp"
    if optimizer_type == "sgd":
        optimize = torch.optim.SGD( params=model.parameters(),
                                    lr = 0.125,
                                    momentum = 0.0,
                                    weight_decay = 12)
        
    elif optimizer_type == "adam":
        optimize = torch.optim.Adam( params=model.parameters(),
                                     lr = 0.0125,
                                     betas = (0.9, 0.999) )
        
    elif optimizer_type == "sgdp":
        optimize = SGDP( params=model.parameters(),
                         lr = 0.1,
                         weight_decay = 1e-5,
                         momentum = 0.9,
                         nesterov = True )

    elif optimizer_type == "adamp":

        optimize = AdamP(   params = model.parameters(),
                            lr = 0.001,
                            betas = (0.9, 0.999),
                            weight_decay = 1e-2,)
    else:
        print("Not support optimizer")
        return

    trainer = Trainer(data, model, optimize, config)

    trainer.train()


if __name__ == '__main__':

    config = {
                "train": {
                        "use_cuda": False,
                        "epochs": 1,
                        "half_lr": True,
                        "early_stop": True,
                        "max_norm": 5,
                    },

                "save_load":
                    {
                        "save_folder": "./checkpoint/",
                        "checkpoint": True,
                        "continue_from": "",
                        "model_path": "final.path.tar",
                    },

                "logging":
                    {
                        "print_freq": 1,
                    }
            }
    # Run the main Function
    main(config)
