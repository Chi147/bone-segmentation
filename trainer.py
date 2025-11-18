from args import get_args
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from utils import dice_loss_from_logits, dice_score_from_logits

def train_model(model, train_loader, val_loader,device):
    args = get_args()
    
    bce = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    for epoch in range(args.epochs):
        model.train()
        running_loss=0

        for image,mask in train_loader:
            images = image.to(device)
            masks = mask.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss_bce = bce(outputs,masks)

            loss_dice =dice_loss_from_logits(outputs,masks)
            loss=loss_bce + loss_dice


            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        if (epoch + 1) % 5 == 0:
            torch.save(model.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")


        train_loss = running_loss / len(train_loader)
            
        val_loss, val_score= validate_model(model, val_loader, bce, device)

        print(f"Epoch {epoch+1}/{args.epochs} |Train loss {train_loss: .4f} |Val loss {val_loss: .4f} Val score {val_score: .4f}")

def validate_model(model, val_loader, loss_fn, device):
    model.eval()
    val_loss = 0.0
    val_score = 0.0

    with torch.no_grad():
        for image,mask in val_loader:
            images = image.to(device)
            masks = mask.to(device)
            outputs = model(images)
            loss_bce = loss_fn(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            val_loss += loss.item()
            val_score += dice_score_from_logits(outputs, masks)


        val_epoch_loss = val_loss / len(val_loader)
        val_epoch_dice = val_score/ len(val_loader)

    return val_epoch_loss, val_epoch_dice