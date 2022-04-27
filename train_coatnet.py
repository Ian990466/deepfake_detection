import argparse
import os
import shutil
from statistics import mode
import warnings

import albumentations as A
import numpy as np
import pandas as pd
from parso import parse
import torch
import torch.multiprocessing
import torchvision
from torchvision import transforms
from torchvision.transforms import ToPILImage, ToTensor

from isplutils import utils, split

torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import roc_auc_score
from tensorboardX import SummaryWriter
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import ImageChops, Image
import matplotlib.pyplot as plt

from architectures.coatnet import coatnet_0
from isplutils.data import FrameFaceIterableDataset, load_face

def tb_attention(tb: SummaryWriter,
                 tag: str,
                 iteration: int,
                 net: nn.Module,
                 device: torch.device,
                 patch_size_load: int,
                 face_crop_scale: str,
                 val_transformer: A.BasicTransform,
                 root: str,
                 record: pd.Series,
                 ):
    # Crop face
    sample_t = load_face(record=record, root=root, size=patch_size_load, scale=face_crop_scale,
                         transformer=val_transformer)
    sample_t_clean = load_face(record=record, root=root, size=patch_size_load, scale=face_crop_scale,
                               transformer=ToTensorV2())
    if torch.cuda.is_available():
        sample_t = sample_t.cuda(device)
    # Transform
    # Feed to net
    with torch.no_grad():
        att: torch.Tensor = net.get_attention(sample_t.unsqueeze(0))[0].cpu()
    att_img: Image.Image = ToPILImage()(att)
    sample_img = ToPILImage()(sample_t_clean)
    att_img = att_img.resize(sample_img.size, resample=Image.NEAREST).convert('RGB')
    sample_att_img = ImageChops.multiply(sample_img, att_img)
    sample_att = ToTensor()(sample_att_img)
    tb.add_image(tag=tag, img_tensor=sample_att, global_step=iteration)

def batch_forward(net: nn.Module, device: torch.device, criterion, data: torch.Tensor, labels: torch.Tensor) -> (torch.Tensor, float, int):
    data = data.to(device)
    labels = labels.to(device)
    out = net(data)
    pred = torch.sigmoid(out).detach().cpu().numpy()
    loss = criterion(out, labels)
    return loss, pred

def validation_routine(net, device, val_loader, criterion, tb, iteration, tag: str, loader_len_norm: int = None):
    net.eval()
    loader_len_norm = loader_len_norm if loader_len_norm is not None else val_loader.batch_size
    val_num = 0
    val_loss = 0.
    pred_list = list()
    labels_list = list()
    for val_data in tqdm(val_loader, desc='Validation', leave=False, total=len(val_loader) // loader_len_norm):
        batch_data, batch_labels = val_data

        val_batch_num = len(batch_labels)
        labels_list.append(batch_labels.flatten())
        with torch.no_grad():
            val_batch_loss, val_batch_pred = batch_forward(net, device, criterion, batch_data,
                                                           batch_labels)
        pred_list.append(val_batch_pred.flatten())
        val_num += val_batch_num
        val_loss += val_batch_loss.item() * val_batch_num

    # Logging
    val_loss /= val_num
    tb.add_scalar('{}/loss'.format(tag), val_loss, iteration)

    if isinstance(criterion, nn.BCEWithLogitsLoss):
        val_labels = np.concatenate(labels_list)
        val_pred = np.concatenate(pred_list)
        val_roc_auc = roc_auc_score(val_labels, val_pred)
        tb.add_scalar('{}/roc_auc'.format(tag), val_roc_auc, iteration)
        tb.add_pr_curve('{}/pr'.format(tag), val_labels, val_pred, iteration)

    return val_loss

def save_model(net: nn.Module, optimizer: optim.Optimizer,
               train_loss: float, val_loss: float,
               iteration: int, batch_size: int, epoch: int,
               path: str):
    path = str(path)
    state = dict(net=net.state_dict(),
                 opt=optimizer.state_dict(),
                 train_loss=train_loss,
                 val_loss=val_loss,
                 iteration=iteration,
                 batch_size=batch_size,
                 epoch=epoch)
    torch.save(state, path)

def main():
    parse = argparse.ArgumentParser(formatter_class= argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--model_type', '-mt', type= str, default= "EfficientNetv2")
    parse.add_argument('--traindb', type=str, help='Training datasets', nargs='+', choices=split.available_datasets, required=True)
    parse.add_argument('--valdb', type=str, help='Validation datasets', nargs='+', choices=split.available_datasets, required=True)
    
    parse.add_argument('--epoch', '-ep', type= int, default= 100)
    parse.add_argument('--ffpp_faces_df_path', type=str, action='store',
                        help='Path to the Pandas Dataframe obtained from extract_faces.py on the FF++ dataset. '
                             'Required for training/validating on the FF++ dataset.')
    parse.add_argument('--ffpp_faces_dir', type=str, action='store',
                        help='Path to the directory containing the faces extracted from the FF++ dataset. '
                             'Required for training/validating on the FF++ dataset.')
    parse.add_argument('--face', type=str, help='Face crop or scale', required=True, choices=['scale', 'tight'])
    parse.add_argument('--size', type=int, help='Train patch size', required=True)

    parse.add_argument('--batch', type=int, help='Batch size to fit in GPU memory', default=32)
    parse.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    parse.add_argument('--valint', type=int, help='Validation interval (iterations)', default= 500)
    parse.add_argument('--patience', type=int, help='Patience before dropping the LR [validation intervals]', default=10)
    parse.add_argument('--maxiter', type=int, help='Maximum number of iterations', default= 1000000)
    parse.add_argument('--init', type=str, help='Weight initialization file')
    parse.add_argument('--scratch', action='store_true', help='Train from scratch')

    parse.add_argument('--trainsamples', type=int, help='Limit the number of train samples per epoch', default=-1)
    parse.add_argument('--valsamples', type=int, help='Limit the number of validation samples per epoch', default= 25000)

    parse.add_argument('--logint', type=int, help='Training log interval (iterations)', default= 500)
    parse.add_argument('--workers', type=int, help='Num workers for data loaders', default=4)
    parse.add_argument('--device', type=int, help='GPU device id', default=0)
    parse.add_argument('--seed', type=int, help='Random seed', default=0)

    parse.add_argument('--debug', action='store_true', help='Activate debug')
    parse.add_argument('--suffix', type=str, help='Suffix to default tag')

    parse.add_argument('--attention', action='store_true', help='Enable Tensorboard log of attention masks')
    parse.add_argument('--log_dir', type=str, help='Directory for saving the training logs', default='runs/binclass/')
    parse.add_argument('--models_dir', type=str, help='Directory for saving the models weights', default='/work/u1240811/weights/CoatNet/binclass/')

    # Model Type
    model = coatnet_0()

    args = parse.parse_args()

    # Required document paths & settings
    model_name = args.model_type
    ffpp_df_path = args.ffpp_faces_df_path
    ffpp_faces_dir = args.ffpp_faces_dir
    train_datasets = args.traindb
    val_datasets = args.valdb

    max_train_samples = args.trainsamples
    max_val_samples = args.valsamples

    num_workers = args.workers
    log_interval = args.logint
    validation_interval = args.valint

    # Hyper parameter
    n_epochs = args.epoch
    batch_size = args.batch
    max_num_iterations = args.maxiter

    # Faces Polices
    face_policy = args.face
    face_size = args.size

    suffix = args.suffix
    debug = args.debug

    # Folders
    logs_folder = args.log_dir
    weights_folder = args.models_dir

    initial_lr = 1e-5
    patience = 10
    seed = 41
    device = 0
    enable_attention = args.attention

    # Random initialization
    np.random.seed(seed)
    torch.random.manual_seed(seed)

    # Put model on GPU & cuDNN nn model optimzation
    torch.backends.cudnn.benchmark= True
    model.to(device)

    tag = utils.make_train_tag(
        net_class= type(model), traindb= train_datasets, face_policy= face_policy,
        patch_size= face_size, seed= seed, suffix= suffix, debug= debug,
        )
    
    # Model checkpoint paths
    bestval_path = os.path.join(weights_folder, tag, 'bestval.pth')
    last_path = os.path.join(weights_folder, tag, 'last.pth')
    periodic_path = os.path.join(weights_folder, tag, 'it{:06d}.pth')

    os.makedirs(os.path.join(weights_folder, tag), exist_ok=True)

    val_loss = min_val_loss = 10
    epoch = iteration = 0
    net_state = None
    opt_state = None

    criterion = nn.BCEWithLogitsLoss()
    min_lr = initial_lr * 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr = initial_lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode= 'min',
        factor= 0.1,
        patience= patience,
        cooldown=2 * patience,
        min_lr=min_lr,
    )

    # Initialize Tensorboard
    logdir = os.path.join(logs_folder, tag)
    if iteration == 0:
        # If training from scratch or initialization remove history if exists
        shutil.rmtree(logdir, ignore_errors=True)

    # TensorboardX instance
    tb = SummaryWriter(logdir=logdir)
    if iteration == 0:
        dummy = torch.randn((1, 3, face_size, face_size), device=device)
        dummy = dummy.to(device)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tb.add_graph(model, [dummy, ], verbose=False)

    transformer = utils.get_transformer(
        face_policy= face_policy,
        patch_size= face_size,
        net_normalizer= transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        train= True
        )


    # Datasets and data loaders
    print('Loading data')
    # Check if paths for DFDC and FF++ extracted faces and DataFrames are provided
    for dataset in train_datasets:
        # if dataset.split('-')[0] == 'dfdc' and (dfdc_df_path is None or dfdc_faces_dir is None):
        #     raise RuntimeError('Specify DataFrame and directory for DFDC faces for training!')
        if dataset.split('-')[0] == 'ff' and (ffpp_df_path is None or ffpp_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for FF++ faces for training!')
    for dataset in val_datasets:
        # if dataset.split('-')[0] == 'dfdc' and (dfdc_df_path is None or dfdc_faces_dir is None):
        #     raise RuntimeError('Specify DataFrame and directory for DFDC faces for validation!')
        if dataset.split('-')[0] == 'ff' and (ffpp_df_path is None or ffpp_faces_dir is None):
            raise RuntimeError('Specify DataFrame and directory for FF++ faces for validation!')
    # Load splits with the make_splits function
    dfdc_df_path = None
    dfdc_faces_dir = None
    splits = split.make_splits(dfdc_df=dfdc_df_path, ffpp_df=ffpp_df_path, dfdc_dir=dfdc_faces_dir, ffpp_dir=ffpp_faces_dir,
                               dbs={'train': train_datasets, 'val': val_datasets})
    train_dfs = [splits['train'][db][0] for db in splits['train']]
    train_roots = [splits['train'][db][1] for db in splits['train']]
    val_roots = [splits['val'][db][1] for db in splits['val']]
    val_dfs = [splits['val'][db][0] for db in splits['val']]

    train_dataset = FrameFaceIterableDataset(roots=train_roots,
                                             dfs=train_dfs,
                                             scale=face_policy,
                                             num_samples=max_train_samples,
                                             transformer=transformer,
                                             size=face_size,
                                             )

    val_dataset = FrameFaceIterableDataset(roots=val_roots,
                                           dfs=val_dfs,
                                           scale=face_policy,
                                           num_samples=max_val_samples,
                                           transformer=transformer,
                                           size=face_size,
                                           )

    print(batch_size)
    train_loader = DataLoader(train_dataset, num_workers=num_workers, batch_size=batch_size, )

    val_loader = DataLoader(val_dataset, num_workers=num_workers, batch_size=batch_size, )

    print('Training samples: {}'.format(len(train_dataset)))
    print('Validation samples: {}'.format(len(val_dataset)))

    if len(train_dataset) == 0:
        print('No training samples. Halt.')
        return

    if len(val_dataset) == 0:
        print('No validation samples. Halt.')
        return

    stop = False
    while not stop:

        # Training
        optimizer.zero_grad()

        train_loss = train_num = 0
        train_pred_list = []
        train_labels_list = []
        for train_batch in tqdm(train_loader):
            model.train()
            batch_data, batch_labels = train_batch

            # print(batch_data)
            # print(batch_labels)
            train_batch_num = len(batch_labels)
            train_num += train_batch_num
            train_labels_list.append(batch_labels.numpy().flatten())

            train_batch_loss, train_batch_pred = batch_forward(model, device, criterion, batch_data, batch_labels)
            train_pred_list.append(train_batch_pred.flatten())

            if torch.isnan(train_batch_loss):
                raise ValueError('NaN loss')

            train_loss += train_batch_loss.item() * train_batch_num

            # Optimization
            train_batch_loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Logging
            if iteration > 0 and (iteration % log_interval == 0):
                train_loss /= train_num
                tb.add_scalar('train/loss', train_loss, iteration)
                tb.add_scalar('lr', optimizer.param_groups[0]['lr'], iteration)
                tb.add_scalar('epoch', epoch, iteration)

                # Checkpoint
                save_model(model, optimizer, train_loss, val_loss, iteration, batch_size, epoch, last_path)
                train_loss = train_num = 0

            # Validation
            if iteration > 0 and (iteration % validation_interval == 0):

                # Model checkpoint
                save_model(model, optimizer, train_loss, val_loss, iteration, batch_size, epoch,
                           periodic_path.format(iteration))

                # Train cumulative stats
                train_labels = np.concatenate(train_labels_list)
                train_pred = np.concatenate(train_pred_list)
                train_labels_list = []
                train_pred_list = []

                train_roc_auc = roc_auc_score(train_labels, train_pred)
                tb.add_scalar('train/roc_auc', train_roc_auc, iteration)
                tb.add_pr_curve('train/pr', train_labels, train_pred, iteration)

                # Validation
                val_loss = validation_routine(model, device, val_loader, criterion, tb, iteration, 'val')
                tb.flush()

                # LR Scheduler
                scheduler.step(val_loss)

                # Model checkpoint
                if val_loss < min_val_loss:
                    min_val_loss = val_loss
                    save_model(model, optimizer, train_loss, val_loss, iteration, batch_size, epoch, bestval_path)

                # Attention
                if enable_attention and hasattr(model, 'get_attention'):
                    model.eval()
                    # For each dataframe show the attention for a real,fake couple of frames
                    for df, root, sample_idx, tag in [
                        (train_dfs[0], train_roots[0], train_dfs[0][train_dfs[0]['label'] == False].index[0],
                         'train/att/real'),
                        (train_dfs[0], train_roots[0], train_dfs[0][train_dfs[0]['label'] == True].index[0],
                         'train/att/fake'),
                    ]:
                        record = df.loc[sample_idx]
                        tb_attention(tb, tag, iteration, model, device, face_size, face_policy,
                                     transformer, root, record)

                if optimizer.param_groups[0]['lr'] == min_lr:
                    print('Reached minimum learning rate. Stopping.')
                    stop = True
                    break

            iteration += 1

            if iteration > max_num_iterations:
                print('Maximum number of iterations reached')
                stop = True
                break

            # End of iteration

        epoch += 1

    # Needed to flush out last events
    tb.close()

    print('Completed')

if __name__ == '__main__':
    main()