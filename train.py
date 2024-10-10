import numpy as np
import torch
import argparse
import os
# distributed training
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DataParallel
from torch.nn.parallel import DistributedDataParallel as DDP

from modules.dataset_loader import video_data_feat
from modules.RMViTModel import RMViTModel
from modules.Loss import Contrastive_loss
from modules.configure_optimizers import configure_optimizers
from model_io import save_model
from modules.sync_batchnorm import convert_model
import time
import datetime
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
torch.multiprocessing.set_sharing_strategy('file_system')
    
def train(args, train_loader_syn, \
          model, criterion, optimizer, scaler, scheduler = None):
    loss_epoch = 0
    model.train()
    
    for step,(syn_i1, syn_i2, quality_label, content_label) in \
    enumerate(train_loader_syn):
 
        x_i1 = syn_i1.cuda(non_blocking=True)
        x_i2 = syn_i2.cuda(non_blocking=True)
        
        x_i1 = x_i1.squeeze(1)
        x_i2 = x_i2.squeeze(1)
        
        # quality classes
        quality_label = torch.zeros((args.batch_size, \
                                  args.quality_clusters+(args.batch_size*args.nodes)))
        quality_label[:args.batch_size,:args.quality_clusters] = quality_label.clone()
        quality_label = quality_label.cuda(non_blocking=True)

        # # same content classes 
        content_label = torch.zeros((args.batch_size, \
                                  args.content_clusters))
        content_label[:args.batch_size,:args.content_clusters] = content_label.clone()
        content_label = content_label.cuda(non_blocking=True)
        
        
        with torch.cuda.amp.autocast(enabled=True):
           z_i, z_j, z_i_patch, z_j_patch, z_i_predict, z_i_T, h_i, h_i_flatten = model(x_i1,x_i2)
           loss = criterion(z_i, z_j,z_i_patch, z_j_patch, z_i_predict, z_i_T, quality_label, content_label)

           if torch.isnan(loss):
              raise ValueError("Loss is NaN. Stopping training.")
        
        # update model weights
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if dist.is_available() and dist.is_initialized():
            loss = loss.data.clone()
            dist.all_reduce(loss.div_(dist.get_world_size()))

        
        if args.nr == 0 and step % 20 == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"Step [{step}/{args.steps}]\t Loss: {loss.item()}\t LR: {round(lr, 5)}")

        if args.nr == 0:
            args.global_step += 1
        loss_epoch += loss.item()
    
    return loss_epoch

def main(gpu, args):
    rank = args.nr * args.gpus + gpu
    
    if args.nodes > 1:
        cur_dir = 'file://' + os.getcwd() + '/sharedfile'
        dist.init_process_group("nccl", init_method=cur_dir,\
                                rank=rank, timeout = datetime.timedelta(seconds=3600),\
                                world_size=args.world_size)
        torch.cuda.set_device(gpu)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    train_dataset = video_data_feat(file_path=args.csv_file_syn)

    if args.nodes > 1:
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True
        )
    else:
        train_sampler = None

    train_loader_syn = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=(train_sampler is None),
        drop_last=True,
        num_workers=args.workers,
        sampler=train_sampler,
    )
    
    args.n_features = 2048
    temporal_model = RMViTModel( num_mem_token=args.num_memory,emb_dim =2048, segment_size=args.size_segment, projection_dim=args.projection_dim, normalize= args.normalize)
    # initialize model
    if args.reload:
        model_fp = os.path.join(
            args.model_path, "checkpoint_{}.tar".format(args.start_epoch-1)
        )
        temporal_model.load_state_dict(torch.load(model_fp, map_location=args.device.type))
    temporal_model = temporal_model.to(args.device)
    
    #sgd optmizer
    args.steps = len(train_loader_syn)
    args.lr_schedule = 'decay'
    args.warmup = args.warmup
    args.weight_decay = args.lr

    args.iters = args.epochs
    optimizer, scheduler = configure_optimizers(args, temporal_model, cur_iter= args.start_epoch-1)
    
    criterion = Contrastive_loss(args.batch_size, args.temperature, args.device, args.world_size)
    
    # DDP / DP
    if args.dataparallel:
        temporal_model = convert_model(temporal_model)
        temporal_model = DataParallel(temporal_model)
        
    else:
        if args.nodes > 1:
            temporal_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(temporal_model)
            temporal_model = DDP(temporal_model, device_ids=[gpu]);print(rank);dist.barrier()

    temporal_model = temporal_model.to(args.device)
    
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    
#    writer = None
    if args.nr == 0:
        print('Training Started')
    
    if not os.path.isdir(args.model_path):
        os.mkdir(args.model_path)
        
    epoch_losses = []
    args.global_step = 0
    args.current_epoch = args.start_epoch
    for epoch in range(args.start_epoch, args.epochs):
        start = time.time()
        loss_epoch = train(args, train_loader_syn, \
          temporal_model, criterion, optimizer, scaler, scheduler)
        
        
        end = time.time()
        print(np.round(end - start,4))
        
        if args.nr == 0 and epoch % 1 == 0:
            save_model(args, temporal_model, optimizer)
            
            torch.save({'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict()},\
                        args.model_path + 'optimizer.tar')
        
        if args.nr == 0:
            print(
                f"Epoch [{epoch}/{args.epochs}]\t Loss: {loss_epoch / args.steps}"
            )
            args.current_epoch += 1
            epoch_losses.append(loss_epoch / args.steps)
            np.save(args.model_path + 'losses.npy',epoch_losses)
        if scheduler:
            scheduler.step()
        
    ## end training
    save_model(args, temporal_model, optimizer)

def parse_args():
    parser = argparse.ArgumentParser(description="RMT-BVQA")
    parser.add_argument('--nodes', type=int, default = 1, help = 'number of nodes', metavar='')
    parser.add_argument('--nr', type=int, default = 0, help = 'rank', metavar='')
    parser.add_argument('--csv_file_syn', type = str, \
                        default = 'csv_files/labels.csv',\
                            help = 'list of filenames of videos with quality and content labels')
    parser.add_argument('--batch_size', type=int, default = 128, \
                        help = 'number of videos in a batch')
    parser.add_argument('--workers', type = int, default = 4, \
                        help = 'number of workers')
    parser.add_argument('--opt', type = str, default = 'sgd',\
                        help = 'optimizer type')
    parser.add_argument('--lr', type = float, default = 0.001,\
                        help = 'learning rate')
    parser.add_argument('--warm', type = float, default = 0.01,\
                        help = 'warmup')
    parser.add_argument('--model_path', type = str, default = 'checkpoints/',\
                        help = 'folder to save trained models')
    parser.add_argument('--temperature', type = float, default = 0.1,\
                        help = 'temperature parameter')
    parser.add_argument('--quality_clusters', type = int, default = 19,\
                        help = 'number of quality classes')
    parser.add_argument('--content_clusters', type = int, default =  149,\
                        help = 'number of content classes')
    parser.add_argument('--reload', type = bool, default = False,\
                        help = 'reload trained model')
    parser.add_argument('--normalize', type = bool, default = True,\
                        help = 'normalize encoder output')
    parser.add_argument('--projection_dim', type = int, default = 128,\
                        help = 'dimensions of the output feature from projector')
    parser.add_argument('--num_memory', type=int, default = 12, \
                        help = 'number of videos in a batch')
    parser.add_argument('--size_seg', type=int, default = 4, \
                        help = 'number of videos in a batch')
    parser.add_argument('--dataparallel', type = bool, default = False,\
                        help = 'use dataparallel module of PyTorch')
    parser.add_argument('--start_epoch', type = int, default = 0,\
                        help = 'starting epoch number')
    parser.add_argument('--end_num', type = int, default = 100,\
                        help = 'number to calculate learning rate decay')
    parser.add_argument('--epochs', type = int, default = 150,\
                        help = 'total number of epochs')
    parser.add_argument('--seed', type = int, default = 10,\
                        help = 'random seed')
    args = parser.parse_args()
    
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.num_gpus = torch.cuda.device_count()
    args.gpus = 1
    args.world_size = args.gpus * args.nodes
    return args

if __name__ == "__main__":
    args = parse_args() 
    if args.nodes > 1:
        print(
            f"Training with {args.nodes} nodes, waiting until all nodes join before starting training"
        )
        mp.spawn(main, args=(args,), nprocs=args.gpus, join=True)
    else:
        main(0, args)