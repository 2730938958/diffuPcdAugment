import torch
import torch.nn as nn
import yaml
import glob
import scipy.io as scio
import random
import numpy as np
import os
from tqdm import tqdm
from evaluate import error
from mmfi import make_dataset, make_dataloader
from lidar_point_transformer import PointTransformerReg

def collate_fn_padd(batch):
    '''
    Padds batch of variable length

    note: it converts things ToTensor manually here since the ToTensor transform
    assume it takes in images rather than arbitrary tensors.
    '''
    ## get sequence lengths
    # for t in batch:
        # print(t['output'].type)
    #     print(a)
        # print(t[0].shape,t[1].shape)
    kpts = []
    [kpts.append(np.array(t['output'])) for t in batch]
    kpts = torch.FloatTensor(np.array(kpts))

    lengths = torch.tensor([t['input_lidar'].shape[0] for t in batch ])
    ## padd
    batch = [torch.Tensor(np.array(t['input_lidar'])) for t in batch ]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    ## compute mask
    batch = batch.permute(1,0,2)
    mask = (batch != 0)

    return batch, kpts, lengths, mask


### dataset
dataset_root = '/hy-tmp/merged'
with open('config.yaml', 'r') as fd:
    config = yaml.load(fd, Loader=yaml.FullLoader)

train_dataset, val_dataset = make_dataset(dataset_root, config)

rng_generator = torch.manual_seed(config['init_rand_seed'])
train_loader = make_dataloader(train_dataset, is_training=True, generator=rng_generator, **config['loader'])
val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['loader'])

## examine dataset
# for i,data in enumerate(train_loader):
#     inputs, labels = data['input_lidar'], data['output']
#     print(inputs.shape)
#     print(labels.shape)
#     if i == 2:
#         break

### model
with open('model_config.yaml', 'r') as fd:
        model_cfg = yaml.load(fd, Loader=yaml.FullLoader)
lidar_cfg = model_cfg['lidar']
model = PointTransformerReg(lidar_cfg)


### training
def test(model, tensor_loader, criterion1, criterion2, device):
    model.eval()
    test_mpjpe = 0
    test_pampjpe = 0
    test_mse = 0
    for data in tqdm(tensor_loader):
        inputs, labels = data['input_lidar'], data['output']
        inputs = inputs.to(device)
        labels.to(device)
        labels = labels.type(torch.FloatTensor)
        outputs = model(inputs)
        outputs = outputs.type(torch.FloatTensor)
        outputs.to(device)
        test_mse += criterion1(outputs,labels).item() * inputs.size(0)

        outputs = outputs.detach().numpy()
        labels = labels.detach().numpy()
        
        mpjpe, pampjpe = criterion2(outputs,labels)
        test_mpjpe += mpjpe.item() * inputs.size(0)
        test_pampjpe += pampjpe.item() * inputs.size(0)
    test_mpjpe = test_mpjpe/len(tensor_loader.dataset)
    test_pampjpe = test_pampjpe/len(tensor_loader.dataset)
    test_mse = test_mse/len(tensor_loader.dataset)
    print("mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format(float(test_mse), float(test_mpjpe),float(test_pampjpe)))
    return test_mpjpe

def train(model, train_loader, test_loader, num_epochs, learning_rate, train_criterion, test_criterion, device):
    optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[20,40],gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,milestones=[15,35],gamma=0.1)
    parameter_dir = './pre-train_weights/best_point_transformers.pth'
    best_test_mpjpe = 100
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        for data in tqdm(train_loader):
            inputs, labels = data['input_lidar'], data['output']
            inputs = inputs.to(device)
            labels = labels.to(device)
            labels = labels.type(torch.FloatTensor)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            outputs = outputs.type(torch.FloatTensor)
            loss = train_criterion(outputs,labels)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * inputs.size(0)
        epoch_loss = epoch_loss/len(train_loader.dataset)
        print('Epoch: {}, Loss: {:.8f}'.format(epoch, epoch_loss))
        if (epoch+1) % 5 == 0:
            test_mpjpe = test(
                model=model,
                tensor_loader=test_loader,
                criterion1 = train_criterion,
                criterion2 = test_criterion,
                device= device
            )
            if test_mpjpe <= best_test_mpjpe:
                print(f"best test mpjpe is:{test_mpjpe}")
                best_test_mpjpe = test_mpjpe
                torch.save(model.state_dict(), parameter_dir)
        scheduler.step()
    return

if not os.path.exists('./pre-train_weights'):
    os.mkdir('pre-train_weights')
train_criterion = nn.MSELoss()
test_criterion = error
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.load_state_dict(torch.load('./pre-train_weights/best_point_transformers.pth'))
model.to(device)
train(
    model = model,
    train_loader= train_loader,
    test_loader= val_loader,
    num_epochs= 50,
    learning_rate=1e-2,
    train_criterion = train_criterion,
    test_criterion = test_criterion,
    device=device
        )
