import torch
import torch.nn as nn
import yaml
import numpy as np
from tqdm import tqdm
from mmfi_lib.evaluate import error
from mmfi_lib.mmfi import make_dataset, make_dataloader
from model.lidar_point_transformer import PointTransformerReg

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
dataset_root = 'D:\\AI\\merged'
with open('configs/config.yaml', 'r') as fd:
    config = yaml.load(fd, Loader=yaml.FullLoader)

train_dataset, val_dataset = make_dataset(dataset_root, config)

rng_generator = torch.manual_seed(config['init_rand_seed'])
val_loader = make_dataloader(val_dataset, is_training=False, generator=rng_generator, **config['loader'])


### model
with open('configs/model_config.yaml', 'r') as fd:
        model_cfg = yaml.load(fd, Loader=yaml.FullLoader)
lidar_cfg = model_cfg['lidar']
model = PointTransformerReg(lidar_cfg)



def eval_entry(model, tensor_loader, criterion1, criterion2, device):
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
        test_mse += criterion1(outputs, labels).item() * inputs.size(0)

        outputs = outputs.detach().numpy()
        labels = labels.detach().numpy()
        
        mpjpe, pampjpe = criterion2(outputs, labels)
        test_mpjpe += mpjpe.item() * inputs.size(0)
        test_pampjpe += pampjpe.item() * inputs.size(0)
    test_mpjpe = test_mpjpe/len(tensor_loader.dataset)
    test_pampjpe = test_pampjpe/len(tensor_loader.dataset)
    test_mse = test_mse/len(tensor_loader.dataset)
    print("mse: {:.8f}, mpjpe: {:.8f}, pampjpe: {:.8f}".format(float(test_mse), float(test_mpjpe),float(test_pampjpe)))
    return test_mpjpe



mse = nn.MSELoss()
e = error
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.load_state_dict(torch.load('./pre-train_weights/lidar_best_point_transformers.pth'))
model.to(device)
eval_entry(
    model=model,
    tensor_loader=val_loader,
    criterion1=mse,
    criterion2=e,
    device=device
        )