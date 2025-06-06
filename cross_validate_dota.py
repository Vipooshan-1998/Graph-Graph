import torch
import numpy as np
from models import *
from dataset_dota import *
from torch.utils.data import DataLoader, Subset

import argparse

import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.optim.lr_scheduler import MultiStepLR

from torchmetrics.functional import pairwise_cosine_similarity
import scipy.io as io

import sklearn
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import torch
import numpy as np
from models import *
from dataset_dota import *
from torch.utils.data import DataLoader

import argparse

import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.optim.lr_scheduler import MultiStepLR

from torchmetrics.functional import pairwise_cosine_similarity
import scipy.io as io

import sklearn
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt

import time
from eval_utils import evaluation
import time
from eval_utils import evaluation

import random

torch.manual_seed(0)  # 3407
np.random.seed(0)
random.seed(0)


parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str, default="data/dota/obj_feat", help="Path to extracted objects data")
parser.add_argument("--img_dataset_path", type=str, default="data/dota/i3d_feat", help="Path to I3D feature data")
parser.add_argument("--toas_files_path", type=str, default="data/dota/toas", help="Path to frame of accidents feature data")
parser.add_argument("--obj_mapping_file", type=str, default="data/dota/obj_idx_to_labels.json",
                    help="path to object label mapping file")
parser.add_argument("--split_path", type=str, default="splits_dota/", help="Path to train/test split")
parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
parser.add_argument("--n_folds", type=int, default=5, help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=1, help="Size of each training batch for frames")
parser.add_argument("--video_batch_size", type=int, default=1, help="Size of each training batch for video")
parser.add_argument("--test_video_batch_size", type=int, default=1, help="Size of each test batch for video")
parser.add_argument("--input_dim", type=int, default=4096, help="input dimension")
parser.add_argument("--img_feat_dim", type=int, default=4096, help="input i3d feature dimension")
parser.add_argument("--embedding_dim", type=int, default=256, help="embedding size of the difference")
parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
parser.add_argument("--ref_interval", type=int, default=20, help="Interval size for reference frames")
parser.add_argument("--fps", type=int, default=10, help="Video fps")
parser.add_argument("--n_frames", type=int, default=50, help="number of frames in a video")
parser.add_argument("--ego_dist", type=int, default=1, help="Ego and other vehicles distance")

opt = parser.parse_args()
print(opt)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Classification criterion
cls_criterion = nn.CrossEntropyLoss().to(device)

acc_best_avg_precision = 0
best_ap = -1
acc_best_ttc = 0
best_ttc = 0

def test_model(epoch, model, test_dataloader, fold):
    """ Function to evaluate the model on the test data
    Inputs:
    epoch: training epoch number (0 if it is only testing)
    model: trained model
    test_dataloader: Dataloader for the testset
    """

    global best_ap, best_ttc
    print("")
    model.eval()
    total_correct, total, all_toa = 0, 0, []

    for batch_i, (
    X, edge_index, y_true, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, obj_vis_feat, batch_vec,
    toa) in enumerate(test_dataloader):

        X = X.reshape(-1, X.shape[2])
        img_feat = img_feat.reshape(-1, img_feat.shape[2])
        edge_index = edge_index.reshape(-1, edge_index.shape[2])
        edge_embeddings = edge_embeddings.view(-1, edge_embeddings.shape[-1])
        video_adj_list = video_adj_list.reshape(-1, video_adj_list.shape[2])
        temporal_adj_list = temporal_adj_list.reshape(-1, temporal_adj_list.shape[2])
        y = y_true.reshape(-1)

        obj_vis_feat = obj_vis_feat.reshape(-1, obj_vis_feat.shape[-1]).to(device)
        feat_sim = pairwise_cosine_similarity(obj_vis_feat + 1e-7, obj_vis_feat + 1e-7)
        temporal_edge_w = feat_sim[temporal_adj_list[0, :], temporal_adj_list[1, :]]
        batch_vec = batch_vec.view(-1).long()

        X, edge_index, y, img_feat, video_adj_list = X.to(device), edge_index.to(device), y.to(device), img_feat.to(
            device), video_adj_list.to(device)
        temporal_adj_list, temporal_edge_w, edge_embeddings, batch_vec = temporal_adj_list.to(
            device), temporal_edge_w.to(device), edge_embeddings.to(device), batch_vec.to(device)
        all_toa += [toa.item()]

        with torch.no_grad():
            logits, probs = model(X, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list,
                                  temporal_edge_w, batch_vec)

        pred_labels = probs.argmax(1)

        total_correct += (pred_labels == y).cpu().numpy().sum()
        total += y.shape[0]

        if batch_i == 0:
            all_probs_vid2 = probs[:, 1].cpu().unsqueeze(0)
            all_pred = pred_labels.cpu()
            all_y = y.cpu()  # .unsqueeze(0)
            all_y_vid = torch.max(y).unsqueeze(0).cpu()  # y.cpu() #.unsqueeze(0)
        else:
            all_probs_vid2 = torch.cat((all_probs_vid2, probs[:, 1].cpu().unsqueeze(0)))
            all_pred = torch.cat((all_pred, pred_labels.cpu()))
            all_y = torch.cat((all_y, y.cpu()))
            all_y_vid = torch.cat((all_y_vid, torch.max(y).unsqueeze(0).cpu()))

        # Empty cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Print the avergae precision
    # np.savez('results_vid2.npz', probs=all_probs_vid2.numpy(), labels=all_y_vid.numpy(), toa=all_toa)
    avg_prec, curr_ttc, _ = evaluation(all_probs_vid2.numpy(), all_y_vid.numpy(), all_toa, opt.fps)
    avg_prec = 100 * avg_prec

    # Print the confusion matrix
    cf = confusion_matrix(all_y.numpy(), all_pred.numpy())
    print(cf)

    # class-wise accuracy
    class_recall = cf.diagonal() / cf.sum(axis=1)
    print(np.round(class_recall, 3))

    # Saving checkpoint
    if avg_prec > best_ap:
        best_ap = avg_prec
        best_ttc = curr_ttc
        os.makedirs("model_checkpoints/dota", exist_ok=True)
        torch.save(model.state_dict(), f"model_checkpoints/dota/{model.__class__.__name__}_{fold+1}_{epoch+1}.pth")
        print(f"Saved the model checkpoint - model_checkpoints/dota/{model.__class__.__name__}_{fold+1}_{epoch+1}.pth")
    print("Best Frame avg precision: %.2f%%" % (best_ap))
    print("Best Frame avg precision's TTA: %.2f" % (best_ttc))

    model.train()
    print("")


def train(train_dataloader, test_dataloader, fold):

    # Define network
    model = SpaceTempGoG_detr_dota(input_dim=opt.input_dim, embedding_dim=opt.embedding_dim,
                                  img_feat_dim=opt.img_feat_dim, num_classes=opt.num_classes).to(device)
    print(model)

    model.train()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")

    learning_rate = 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = MultiStepLR(optimizer, milestones=[25], gamma=0.5)

    # Training process
    ttc = 0
    for epoch in range(opt.num_epochs):

        start = time.time()
        epoch_metrics = {"c1_loss": []}

        print(f"--- Epoch {epoch+1} ---")

        loss, all_toa = 0, []

        for batch_i, (
        X, edge_index, y_true, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, obj_vis_feat, batch_vec,
        toa) in enumerate(train_dataloader):

            # Processing the inputs from the dataloader
            X = X.reshape(-1, X.shape[2])
            img_feat = img_feat.reshape(-1, img_feat.shape[2])
            edge_index = edge_index.reshape(-1, edge_index.shape[2])
            edge_embeddings = edge_embeddings.view(-1, edge_embeddings.shape[-1])
            video_adj_list = video_adj_list.reshape(-1, video_adj_list.shape[2])
            temporal_adj_list = temporal_adj_list.reshape(-1, temporal_adj_list.shape[2])
            y = y_true.reshape(-1)
            toa = toa.item()

            # pairwise cosine similarity between the objects
            obj_vis_feat = obj_vis_feat.reshape(-1, obj_vis_feat.shape[-1]).to(device)
            feat_sim = pairwise_cosine_similarity(obj_vis_feat + 1e-7, obj_vis_feat + 1e-7)
            temporal_edge_w = feat_sim[temporal_adj_list[0, :], temporal_adj_list[1, :]]
            batch_vec = batch_vec.view(-1).long()

            X, edge_index, y, img_feat, video_adj_list = X.to(device), edge_index.to(device), y.to(device), img_feat.to(
                device), video_adj_list.to(device)
            temporal_adj_list, temporal_edge_w, edge_embeddings, batch_vec = temporal_adj_list.to(
                device), temporal_edge_w.to(device), edge_embeddings.to(device), batch_vec.to(device)

            # Get predictions from the model
            logits, probs = model(X, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list,
                                  temporal_edge_w, batch_vec)

            # Exclude the actual accident frames from the training
            c_loss1 = cls_criterion(logits[:toa], y[:toa])
            loss = loss + c_loss1

            if (batch_i + 1) % 3 == 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                loss = 0

            pred_labels = probs.argmax(1)
            total_pred = (pred_labels == y).cpu().numpy().sum()

            # Keep track of epoch metrics
            epoch_metrics["c1_loss"].append(c_loss1.item())

            if batch_i == 0:
                all_probs_vid2 = probs[:, 1].detach().cpu().unsqueeze(0)
                all_y_vid = torch.max(y).unsqueeze(0).cpu()  # y.cpu() #.unsqueeze(0)
            else:
                all_probs_vid2 = torch.cat((all_probs_vid2, probs[:, 1].detach().cpu().unsqueeze(0)))
                all_y_vid = torch.cat((all_y_vid, torch.max(y).unsqueeze(0).cpu()))
            all_toa += [toa]

            if batch_i % 100 == 0:
                print("[Fold %d/5] [Epoch %d/%d] [Batch %d/%d] [CE Loss: %f (%f)] [LR : %.6f]"
                      % (
                          fold+1,
                          epoch+1,
                          opt.num_epochs,
                          batch_i,
                          len(train_dataloader),
                          c_loss1.item(),
                          np.mean(epoch_metrics["c1_loss"]),
                          optimizer.param_groups[0]['lr'],
                      ))

            # Empty cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        end = time.time()
        print(f"Time: {end - start}")

        # Print the avergae precision
        _, ttc, _ = evaluation(all_probs_vid2.numpy(), all_y_vid.numpy(), all_toa, opt.fps)

        # Testing the model
        test_model(epoch, model, test_dataloader, fold)


        scheduler.step()

if __name__ == "__main__":
    

    # Define training set
    dataset = CrossValDataset(
        img_dataset_path=opt.img_dataset_path,
        dataset_path=opt.dataset_path,
        toas_files_path=opt.toas_files_path,
        split_path=opt.split_path,
        #		frame_batch_size=opt.batch_size,
        ref_interval=opt.ref_interval,
        objmap_file=opt.obj_mapping_file,
        training=True,
        ego_dist=opt.ego_dist,
        n_frames=opt.n_frames,
    )

    folds = opt.n_folds
    kf = KFold(n_splits=folds, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(dataset)):

        print('running split:', fold+1)

        folder = 'splt_idx'
        os.makedirs(folder, exist_ok=True)

        with open(f"{folder}/fold_{fold+1}_train.txt", "w") as f:
            f.write('\n'.join(map(str, train_idx)))

        with open(f"{folder}/fold_{fold+1}_test.txt", "w") as f:
            f.write('\n'.join(map(str, test_idx)))

        train_dataset = Subset(dataset, train_idx)
        test_dataset = Subset(dataset, test_idx)

        train_dataloader = DataLoader(train_dataset, batch_size=opt.video_batch_size, shuffle=True, num_workers=8)
        test_dataloader = DataLoader(test_dataset, batch_size=opt.test_video_batch_size, shuffle=False, num_workers=8)

        train(train_dataloader, test_dataloader, fold)

        acc_best_avg_precision += best_ap
        acc_best_ttc += best_ttc
        best_ap = -1

    print("average of all best average precision: ", acc_best_avg_precision/5)
    print("average of all best AP's time to collision (mTTA): ", acc_best_ttc/5)

