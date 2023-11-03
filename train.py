import logging
import os
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score
from tabulate import tabulate
import matplotlib.pyplot as plt
from util import measure_cluster, seed_everything, print_network
from experiments import get_experiment_config
from models import AMCFCN
from datatool import load_dataset
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
Bestacc=0
def get_current_labels(train_loader, model, device):
    model.eval()
    labels = []
    for data in train_loader:
        # measure data loading time
        Xs = [d.to(device) for d in data[0]]
        labels.append(model.predict(Xs).detach().cpu())
    labels = torch.cat(labels).long()
    return labels

def train_step(train_loader, model, epoch, device, verbose=1):
    model.train()
    global  n
    tot_losses = []
    con_losses = []
    clu_losses = []
    tot_losses_list = []
    log_dir = os.path.join(args.log_dir, f'run_{run}')    
    if verbose:
        pbar = tqdm(total=len(train_loader), ncols=0, unit=" batch")
    for data in train_loader:
        # measure data loading time
        Xs = [d.to(device) for d in data[0]]
        model.optimizer.zero_grad()
        tot_loss, clu_loss, con_loss = model.get_loss(Xs)
        tot_losses.append(tot_loss.item())
        con_losses.append(con_loss.item())
        clu_losses.append(clu_loss.item())
        tot_loss.backward()
        model.optimizer.step()
        if verbose:
            pbar.update()
            pbar.set_postfix(
                epoch=epoch,
                total_loss=f"{np.mean(tot_losses):.4f}",
                clustering_loss=f"{np.mean(clu_losses):.4f}",
                contrastive_loss=f"{np.mean(con_losses):.4f}",
            )
    if verbose:
        pbar.close()
    tot_loss_mean = np.mean(tot_losses)
    clu_losses_mean = np.mean(clu_losses)
    con_losses_mean = np.mean(con_losses)
    tot_losses_list.append(tot_loss_mean)
    
    return np.mean(tot_losses), np.mean(clu_losses), np.mean(con_losses)

def save_dict(obj, path):
    try:
        with open(path, 'w') as f:
            save_dict = {}
            for key in obj.keys():
                if isinstance(obj[key], list):
                    save_dict[key] = obj[key]
                elif isinstance(obj[key], int):
                    save_dict[key] = obj[key]
                elif isinstance(obj[key], np.ndarray):
                    save_dict[key] = obj[key].tolist()
            json.dump(save_dict, f, indent=4)
            print(f'Saved dict at {path}')
    except Exception as e:
        print(e)


class Recoder:

    def __init__(self):
        self.epoch = []
        self.total_losses = []
        self.contrastive_losses = []
        self.clustering_losses = []
        self.accuracy = []
        self.nmi = []
        

    def batch_update(self, epoch, tot_loss, clu_loss, con_loss, acc, nmi):
        self.epoch.append(epoch)
        self.total_losses.append(tot_loss)
        self.contrastive_losses.append(con_loss)
        self.clustering_losses.append(clu_loss)
        self.accuracy.append(acc)
        self.nmi.append(nmi)
        

    def to_dict(self):
        return {"epoch": self.epoch,
                "total_losses": self.total_losses,
                "contrastive_losses": self.contrastive_losses,
                "clustering_losses": self.clustering_losses,
                "accuracy": self.accuracy,
                "nmi": self.nmi,
               }


def main(model, dataset, args, run):
    ### Data loading ###
    global Bestacc
    num_workers = 8
    model.to(args.device)
    history = Recoder()
    train_loader = DataLoader(dataset, args.batch_size, num_workers=num_workers,
                              shuffle=True, drop_last=True)
    valid_loader = DataLoader(dataset, args.batch_size*2, num_workers=num_workers, shuffle=False)
    valid_loader.transform = None
    log_dir = os.path.join(args.log_dir, f'run_{run}')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    writer = SummaryWriter(log_dir=log_dir)
    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'), level=logging.INFO)
    hparams_head = ['Hyper-parameters', 'Value']
    logging.info(tabulate(args.dict().items(), headers=hparams_head))
    targets = valid_loader.dataset.targets
    previous_label = None
    if isinstance(targets, list):
        targets = np.array(targets)
    elif isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    else:
        raise ValueError('targets must be list, numpy or tensor.')
    best_loss = np.inf
    
    print('please waiting! the net is [TRAIN]...')
    ###############################################################################
   
    for epoch in range(args.epochs):
   
      

        # Supervised Training
        tot_loss_avg, clu_loss_avg, con_loss_avg = \
            train_step(train_loader, model, epoch+1, args.device, verbose=args.verbose)
        
        
        
        if epoch % args.validation_intervals == 0:
            model.eval()
            predicted = get_current_labels(valid_loader, model, args.device).numpy()
            acc, nmi, pur = measure_cluster(predicted, targets)
            if acc> Bestacc:
                Bestacc=acc
                print("acc",Bestacc)
            if previous_label is not None:
                nmi_t_1 = normalized_mutual_info_score(predicted, previous_label)
            else:
                nmi_t_1 = 0
            previous_label = predicted
            if args.verbose:
                values = [(epoch+1, acc, nmi, pur, nmi_t_1)]
                headers = ['Validation Epoch', 'Accuracy', 'NMI', 'Purity', 'nmi_(t-1)']
                print(tabulate(values, headers=headers))
            
        if tot_loss_avg < best_loss:
            torch.save(model.state_dict(), os.path.join(log_dir, f'model_weight_best.pth'))
            best_loss = tot_loss_avg
            print(f"Saved model at {os.path.join(log_dir, f'model_weight_best.pth')}, best loss: {best_loss:.6f}.")

    writer.close()
    return history.to_dict()


if __name__ == '__main__':
    name, args = get_experiment_config()
    seed_everything(args.seed)
    dataset = load_dataset(args.ds_name, args.img_size)
    hparams_head = ['Hyper-parameters', 'Value']
    run_histories = []
    for run in range(args.n_runs):
        model = AMCFCN(args)
        print_network(model)
        history = main(model, dataset, args, run)
        run_histories.append(history)
    if args.extra_record:
        torch.save(run_histories, os.path.join(args.log_dir, 'records.his'))
        logging.info(f"Saved records at {os.path.join(args.log_dir, 'records.his')}")
