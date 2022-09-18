import os
import shutil
import glob
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from translation_model.model import *
from translation_model.loss import *
from tortoise.utils.audio import normalize_tacotron_mel, denormalize_tacotron_mel

EPOCHS = 10000

def collate_fn(batch):
    batch = list(zip(*batch))
    max_1 = max(max([item.shape[1] for item in batch[0]]), max([item.shape[1] for item in batch[2]]))
    max_2 = max([item.shape[0] for item in batch[1]])
    for i in range(len(batch)):
        item_list = batch[i]
        if i != 1:
            pad_idx = 1
            max_size = max_1
        else:
            pad_idx = 0
            max_size = max_2
        # max_size = max([item.shape[pad_idx] for item in item_list])
        if pad_idx == 1:
            item_list = [F.pad(item, (0, max_size - item.shape[pad_idx])) for item in item_list]
        else:
            item_list = [F.pad(item, (0, 0, 0, max_size - item.shape[pad_idx])) for item in item_list]
        if item_list[0].ndim == 2:
            item_list = torch.stack(item_list, dim=0)
        else:
            item_list = torch.cat(item_list, dim=0)

        batch[i] = item_list
    return batch


class MelDataset(torch.utils.data.Dataset):
    def __init__(self):
        super().__init__()
        
        self.train_files = glob.glob('train/mels/inputs/*.pth')[6:9]
        # self.targ_files = glob.glob('train/mel/targs/*.pth')
    
    def __len__(self):
        return len(self.train_files)
    
    def __getitem__(self, idx):
        train_file = self.train_files[idx]
        targ_file = train_file.replace('inputs', 'targs')
        
        train_mel, text_lats, _ = torch.load(train_file, map_location = torch.device('cpu'))
        targ_mel = torch.load(targ_file, map_location = torch.device('cpu'))
        
        # these are all 3d, turn them into 2d
        return train_mel[0], text_lats[0], targ_mel[0]
    

ds = MelDataset()
train_loader = torch.utils.data.DataLoader(ds, batch_size=12, shuffle=True, collate_fn=collate_fn, 
                                           num_workers=4, pin_memory=True)

model = TranslationModel(4)
model.load_state_dict(torch.load('models/basic_4layer_msedouble_700.pth'))
model.cuda()
opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 
                                            max_lr=5e-4, div_factor=100,
                                            total_steps=EPOCHS*len(train_loader), last_epoch=-1, 
                                            pct_start=0.1)

os.makedirs('models', exist_ok=True)
os.makedirs('train_outs', exist_ok=True)
sdtw = SoftDTW(use_cuda=True, gamma=0.1)
TACOTRON_MEL_MAX = 2.3143386840820312
TACOTRON_MEL_MIN = -11.512925148010254

for epoch in range(EPOCHS):
    pbar = tqdm(train_loader)
    for i, (train_mel, text_lats, targ_mel) in enumerate(pbar):
        i = epoch * len(train_loader) + i
        train_mel, text_lats, targ_mel = train_mel.cuda(), text_lats.cuda(), targ_mel.cuda()
        train_mel, targ_mel = normalize_tacotron_mel(train_mel), normalize_tacotron_mel(targ_mel)
        pred_mel_train = model(train_mel, text_lats)
        pred_mel_targ = model(targ_mel, text_lats)
        
        # loss = sdtw(pred_mel.permute(0, 2, 1), targ_mel.permute(0, 2, 1)).mean() * 1e-8
        # import pdb; pdb.set_trace()
        
        # Gpred = torch.bmm(pred_mel_train, pred_mel_train.permute(0,2,1))
        # Gtarg = torch.bmm(targ_mel, targ_mel.permute(0,2,1))
        # gram_loss = 1e-3 * (Gpred - Gtarg).abs().mean()
        gram_loss = 0
        
        content_loss_targ = F.mse_loss(pred_mel_targ, targ_mel)
        content_loss_train = F.mse_loss(pred_mel_train, targ_mel)  # maybe dont use
        
        loss = content_loss_targ + content_loss_train + gram_loss
        
        opt.zero_grad()
        loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        norm = 0
        opt.step()
        sched.step()
    
        pbar.set_description(
            f'Epoch: {epoch+1}, Step: {i}, Gram Loss: {gram_loss:.2f}, Cont Targ: {content_loss_targ:.5f}, Cont Train: {content_loss_train:.5f}'
        )
    
    if epoch%1000 == 999:
        torch.save(model.state_dict(), f'models/basic_4layer_msedouble_overfit_{epoch+1}.pth')
