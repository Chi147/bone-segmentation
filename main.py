from args import get_args
import os
import pandas as pd
import torch
from trainer import train_model
from dataset import Knee_dataset
from torch.utils.data import DataLoader
from model import UNetLext

def main():
    args = get_args()
    
    #Step1: Load csv files
    train_set= pd.read_csv(os.path.join(args.csv_dir, 'train.csv'))
    val_set= pd.read_csv(os.path.join(args.csv_dir, 'val.csv'))

    #Step2: Prepare dataset
    train_dataset = Knee_dataset(train_set)
    val_dataset = Knee_dataset(val_set)

    #Step3: Initializing dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    #Step4: Initializing the model
    model = UNetLext(input_channels=1,
                     output_channels=1,
                     pretrained=False,
                     path_pretrained='',
                     restore_weights=False,
                     path_weights=''                   
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #Step5: Model to device
    model = model.to(device)

    #Step6: Train
    train_model(model, train_loader, val_loader,device)

if __name__ == '__main__':
    main()