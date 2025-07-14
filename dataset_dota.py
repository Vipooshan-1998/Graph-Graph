import glob
import random
import os
import numpy as np
import torch
import math

from torch.utils.data import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
import scipy.io as io
import json

from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset

from nltk.tokenize import sent_tokenize, word_tokenize
import spacy

# import logging
# Configure logging
# logging.basicConfig(level=logging.INFO)  # Set log level

class Dataset(Dataset):
    def __init__(self, dataset_path, img_dataset_path, split_path, ref_interval, objmap_file, training):

        """
        Input:
        dataset_path: Path to the video frames
        img_dataset_path: Path to the I3D frame features
        split_path: Path to the split files
        ref_interval: Number of frames temporally connected in frame-level graph
        objmap_file: The file mapping labels to label names for objects
        annofile: File containing TOA annotations
        training: Flag to determine train/test phase

        """

        self.training = training
        self.img_dataset_path = img_dataset_path
        self.feature_paths = self._extract_feature_paths(dataset_path, split_path, training)
        self.transform = transforms.Compose([transforms.ToTensor(), ])
        # self.frame_batch_size = frame_batch_size
        self.ref_interval = ref_interval
        self.temporal_ref = 1
        self.dilation_factor = 1
        self.topk = 10
        self.frame_stats_path = dataset_path[:-8] + 'frames_stats'  # (height, width)
        self.n_frames = 50

        # Obj label to word embeddings
        self.idx_to_classes_obj = json.load(open(objmap_file))
        self.nlp = spacy.load('en_core_web_md', disable=['ner', 'parser'])
        self.obj_embeddings = torch.from_numpy(np.array([self.nlp(obj).vector for obj in self.idx_to_classes_obj]))

    def _extract_feature_paths(self, dataset_path, split_path="splits_dota/", training=True):

        """ Function to extract paths to frames given the specified train/test split
        Input:
        dataset_path: Path to the video frames
        split_path: Path to the split files
        training: Flag to determine train/test phase

        Returns:
        feature_paths: List of all the video paths in a split

        """
        fn = "train_split.txt" if training else "test_split.txt"
        # fn = "split.txt" if training else "split.txt"
        split_path = os.path.join(split_path, fn)
        with open(split_path) as file:
            lines = file.read().splitlines()
        feature_paths = []

        for line in lines:
            if training:
                feature_paths += [os.path.join(dataset_path, "training", line)]
            else:
                feature_paths += [os.path.join(dataset_path, "testing", line)]
            # if training:
            #     feature_paths += [os.path.join(dataset_path, line)]
            # else:
            #     feature_paths += [os.path.join(dataset_path, line)]
        # print(feature_paths)
        return feature_paths

    def _frame_number(self, feat_path):

        """ Function to extract the frame number from the input
        Input:
        feat_path: Input path for a frame

        Returns:
        Integer of the frame number (temporal position of a frame in a video)

        """
        return int(feat_path.split('/')[-1].split('.mat')[0].split('_')[-1])

    def _get_distances(self, a, b, p):
        return torch.abs(((b[1] - a[1])*p[0] - (b[0] - a[0])*p[1] + b[0]*a[1] - b[1]*a[0]) / torch.sqrt(torch.pow(b[1] - a[1], 2) + torch.pow(b[0] - a[0], 2)))

    def get_toa_all(self, video_name):

        """ 
        get toa from toas directory
        """

        toa_dir = self.frame_stats_path[:-12] + 'toas'
        toa_file = os.path.join(toa_dir, video_name+'.txt')

        with open(toa_file) as f:
            return int(f.read())
        

    def __getitem__(self, index):

        feature_path = self.feature_paths[index % (len(self))]
        video_name = feature_path.split('/')[-1][:-4]
        # print(video_name)

        # Load the data.npy (for features) and det.npy (for bounding boxes) files
        all_data = np.load(f"{feature_path}")
        all_feat = torch.from_numpy(all_data['data'])[:, 1:, :]
        all_bbox = torch.from_numpy(
            all_data['det']).float()  # (x1, y1, x2, y2, cls, accident/no acc)bottom left and top right coordinates

        # curr_vid_label = int(all_data['labels'][1])
        if "neg" in feature_path.split('/')[-1]:
            curr_vid_label = 0
        else:
            curr_vid_label = 1
            
        if curr_vid_label > 0:
            curr_toa = self.get_toa_all(video_name)
        else:
            curr_toa = self.n_frames + 1

        # Reading frame (i3d) features for the frames
        if curr_vid_label > 0:
            img_file = os.path.join(self.img_dataset_path, feature_path.split('/')[-2], "positive",
                                    feature_path.split('/')[-1].split(".")[0] + '.npy')
        else:
            img_file = os.path.join(self.img_dataset_path, feature_path.split('/')[-2], "negative",
                                    feature_path.split('/')[-1].split(".")[0] + '.npy')

        # if curr_vid_label > 0:
        #     img_file = os.path.join(self.img_dataset_path, "positive",
        #                             feature_path.split('/')[-1].split(".")[0] + '.npy')
        # else:
        #     img_file = os.path.join(self.img_dataset_path, "negative",
        #                             feature_path.split('/')[-1].split(".")[0] + '.npy')

        all_img_feat = self.transform(np.load(img_file)).squeeze(0)

        # Reading frame stats file
        if curr_vid_label > 0:
            frame_stats_file = os.path.join(self.frame_stats_path, feature_path.split('/')[-2], "positive",
                                            feature_path.split('/')[-1].split(".")[0] + '.npy')
        else:
            frame_stats_file = os.path.join(self.frame_stats_path, feature_path.split('/')[-2], "negative",
                                            feature_path.split('/')[-1].split(".")[0] + '.npy')
        # if curr_vid_label > 0:
        #     frame_stats_file = os.path.join(self.frame_stats_path, "positive",
        #                                     feature_path.split('/')[-1].split(".")[0] + '.npy')
        # else:
        #     frame_stats_file = os.path.join(self.frame_stats_path, "negative",
        #                                     feature_path.split('/')[-1].split(".")[0] + '.npy')
        
        frame_stats = torch.from_numpy(np.load(frame_stats_file)).float()

        # Calculating the bbox centers
        cx, cy = (all_bbox[:, :, 0] + all_bbox[:, :, 2]) / 2, (all_bbox[:, :, 1] + all_bbox[:, :, 3]) / 2
        all_obj_centers = torch.cat((cx.unsqueeze(2), cy.unsqueeze(2)), 2).float()

        # Obj label indexes
        all_obj_label_idxs = all_bbox[:, :, -1].long()

        # ---------- Building the object-level spatio-temporal graph ---------------
        video_data, edge_weights, edge_embeddings, obj_vis_feat = [], [], [], []
        start_idx = 0
        actual_frame_idx = 0
        obj_refs = {}
        num_objs_list = []

        for i in range(all_feat.shape[0]):

            obj_label_idxs, obj_feat, obj_centers, bbox = all_obj_label_idxs[i], all_feat[i], all_obj_centers[i], all_bbox[i]
            num_objs = len(obj_label_idxs)

            # print()
            # print(obj_label_idxs.shape, obj_feat.shape, obj_centers.shape)

            # Create the adj list (making big disconnected graph for one video)
            # relation pair idxs
            source_nodes = torch.arange(obj_label_idxs.shape[0]).unsqueeze(1).repeat(1,
                                                                                     obj_label_idxs.shape[0]).flatten()
            target_nodes = torch.arange(obj_label_idxs.shape[0]).unsqueeze(0).repeat(1,
                                                                                     obj_label_idxs.shape[0]).flatten()

            # print(source_nodes, target_nodes)
            
            adj_list = torch.cat((source_nodes.unsqueeze(1), target_nodes.unsqueeze(1)), 1)
            repeat_idx = torch.where(adj_list[:, 0] != adj_list[:, 1])[0]  # removing the self loops from the adj list
            adj_list = adj_list[repeat_idx]
            adj_list = torch.LongTensor(adj_list)
            adj_list = adj_list.permute((1, 0))

            # print("adj_list ", adj_list.shape)

            # Edge embeddings - dist_rel, obj centers of source, obj centers of target
            source_nodes, target_nodes = adj_list[0, :], adj_list[1, :]
            obj_centers = obj_centers / frame_stats[i] 
            bbox[:, 0:2] = bbox[:, 0:2] / frame_stats[i]  # normalize with frame height and width
            bbox[:, 2:4] = bbox[:, 2:4] / frame_stats[i]
            source_centers = obj_centers[source_nodes]
            target_centers = obj_centers[target_nodes]

            # print(source_nodes, target_nodes)

            # print(source_centers.shape, target_centers.shape)

            # dist_mat = self.get_distance(source_centers, target_centers).float()
            dist_mat = torch.cdist(source_centers, target_centers).float()

            n = bbox.shape[0]-1

            dist = self._get_distances((bbox[0, 0].unsqueeze(0).repeat(n, 1), bbox[0, 1].unsqueeze(0).repeat(n, 1)), (bbox[0, 2].unsqueeze(0).repeat(n, 1), bbox[0, 3].unsqueeze(0).repeat(n, 1)), ((bbox[1:, 0]+bbox[1:, 2])/2, bbox[1:, 3]))
            dist = torch.nan_to_num(dist, nan=0.0)
            
            # print("dist_mat ", dist_mat1.shape, dist_mat2.shape)
            dist_mat = F.softmax(-dist_mat, dim=-1)
            dist_rel = dist_mat[adj_list[0, :], adj_list[1, :]].unsqueeze(1)

            dist_rel[0:n, :] = dist[0].unsqueeze(1)

            edge_embed = torch.cat((source_centers, target_centers, dist_rel), 1)

            adj_list += start_idx

            target = curr_vid_label  # Target

            # Adding obj label embeddings to the node features
            label_embeddings = self.obj_embeddings[obj_label_idxs]
            x_embed = torch.cat((obj_feat, label_embeddings), 1)
            frame_data = Data(x=x_embed, edge_index=adj_list, y=target)

            video_data.append(frame_data)
            edge_embeddings.append(edge_embed)
            obj_vis_feat.append(obj_feat)

            obj_refs['f_' + str(actual_frame_idx)] = {}
            unique_obj_idxs = torch.unique(obj_label_idxs)
            for obj_ in unique_obj_idxs:
                idxx = torch.where(obj_label_idxs == obj_)[0] + start_idx
                obj_refs['f_' + str(actual_frame_idx)][obj_.item()] = idxx

            start_idx += num_objs

            # keep track of num of objects for pooling to create graph-embedding later
            num_objs_list += [torch.zeros(num_objs) + actual_frame_idx]
            actual_frame_idx += 1

        data, slices = InMemoryDataset.collate(video_data)
        edge_embeddings = torch.cat(edge_embeddings)
        obj_vis_feat = torch.cat(obj_vis_feat)
        num_objs_list = torch.cat(num_objs_list)

        # Generate the temporal connections adjacency matrix for the object-level graph
        temporal_adj_list, temporal_edge_w = [], []
        for i, (key_, value_) in enumerate(obj_refs.items()):
            temporal_ref = min(i, self.temporal_ref)
            curr_frame = obj_refs['f_' + str(i)]
            for t_ in range(1, temporal_ref + 1):
                prev_frame = obj_refs['f_' + str(i - t_)]
                for obj_l, t_n in curr_frame.items():
                    if obj_l in prev_frame:
                        s_n = prev_frame[obj_l]  # source node idxs
                        num_s = s_n.shape[0]
                        # repeat all sources nodes length of target times - for example if source = [1, 2] and len(target)=2 it becomes [1,1,2,2]
                        s_n = s_n.unsqueeze(0).repeat(len(t_n), 1).T.reshape(-1, 1)
                        t_n = t_n.repeat(num_s).unsqueeze(1)  # repeat target source times directly
                        s_t = torch.cat((s_n, t_n), 1)
                        # temporal_edge_w.append(feat_sim[s_t[:, 0], s_t[:, 1]])    #choose the feature similarity as edge weight
                        temporal_adj_list += s_t.tolist()

        temporal_adj_list = torch.Tensor(temporal_adj_list).permute((1, 0)).long()

        # Generate the frame-level adjacency matrix
        video_adj_list = []
        for i in range(len(video_data)):
            bw_ref_interval = min(i, self.ref_interval)
            for j in range(1, bw_ref_interval + 1, self.dilation_factor):
                video_adj_list += [[i - j, i]]  # adding previous ref_interval neighbors
        video_adj_list = torch.Tensor(video_adj_list).permute((1, 0)).long()

        return data.x, data.edge_index, data.y, all_img_feat, video_adj_list, edge_embeddings, temporal_adj_list, obj_vis_feat, num_objs_list, curr_toa

    def __len__(self):
        return len(self.feature_paths)
    
class CrossValDataset(Dataset):
    def __init__(self, dataset_path, img_dataset_path, toas_files_path, split_path, ref_interval, objmap_file, training, ego_dist, n_frames):

        """
        Input:
        dataset_path: Path to the video frames
        img_dataset_path: Path to the I3D frame features
        split_path: Path to the split files
        ref_interval: Number of frames temporally connected in frame-level graph
        objmap_file: The file mapping labels to label names for objects
        annofile: File containing TOA annotations
        training: Flag to determine train/test phase

        """

        self.training = training
        self.img_dataset_path = img_dataset_path
        self.feature_paths = self._extract_feature_paths(dataset_path, split_path, training)
        self.transform = transforms.Compose([transforms.ToTensor(), ])
        # self.frame_batch_size = frame_batch_size
        self.ref_interval = ref_interval
        self.temporal_ref = 1
        self.dilation_factor = 1
        self.toas_files_path = toas_files_path
        self.topk = 10
        self.frame_stats_path = dataset_path[:-8] + 'frames_stats'  # (height, width)
        self.n_frames = n_frames
        self.ego_dist = ego_dist

        # Obj label to word embeddings
        self.idx_to_classes_obj = json.load(open(objmap_file))
        self.nlp = spacy.load('en_core_web_md', disable=['ner', 'parser'])
        self.obj_embeddings = torch.from_numpy(np.array([self.nlp(obj).vector for obj in self.idx_to_classes_obj]))

    def _extract_feature_paths(self, dataset_path, split_path="splits_dota/", training=True):

        """ Function to extract paths to frames given the specified train/test split
        Input:
        dataset_path: Path to the video frames
        split_path: Path to the split files
        training: Flag to determine train/test phase

        Returns:
        feature_paths: List of all the video paths in a split

        """
        fn = "split.txt"
        split_path = os.path.join(split_path, fn)
        with open(split_path) as file:
            lines = file.read().splitlines()
        feature_paths = []

        for line in lines:
            feature_paths += [os.path.join(dataset_path, line)]

        return feature_paths

    def _frame_number(self, feat_path):

        """ Function to extract the frame number from the input
        Input:
        feat_path: Input path for a frame

        Returns:
        Integer of the frame number (temporal position of a frame in a video)

        """
        return int(feat_path.split('/')[-1].split('.mat')[0].split('_')[-1])

    def _get_distances(self, a, b, p):
        return torch.abs(((b[1] - a[1])*p[0] - (b[0] - a[0])*p[1] + b[0]*a[1] - b[1]*a[0]) / torch.sqrt(torch.pow(b[1] - a[1], 2) + torch.pow(b[0] - a[0], 2)))

    def get_toa_all(self, video_name):

        """ 
        get toa from toas directory
        """

        toa_dir = self.toas_files_path
        toa_file = os.path.join(toa_dir, video_name+'.txt')

        with open(toa_file) as f:
            return int(f.read())
        

    def __getitem__(self, index):

        feature_path = self.feature_paths[index % (len(self))]
        video_name = feature_path.split('/')[-1][:-4]

        # Load the data.npy (for features) and det.npy (for bounding boxes) files
        all_data = np.load(f"{feature_path}")
        all_feat = torch.from_numpy(all_data['data'])[:, 1:, :]
        all_bbox = torch.from_numpy(
            all_data['det']).float()  # (x1, y1, x2, y2, cls, accident/no acc)bottom left and top right coordinates

        # curr_vid_label = int(all_data['labels'][1])
        # logging.info("all_data[labels]: ", all_data['labels'])
        # logging.info('curr_vid_label: ', curr_vid_label)
        # logging.info('feature_path.split: ', feature_path.split('/'))
        if "neg" in feature_path.split('/')[-1]:
            curr_vid_label = 0
        else:
            curr_vid_label = 1
        if curr_vid_label > 0:
            curr_toa = self.get_toa_all(video_name)
            # print('---------------TOA------------')
            # print(video_name)
            # print(curr_toa)
        else:
            curr_toa = self.n_frames + 1

        # Reading frame (i3d) features for the frames
        # logging.info('-------------------------------------------------------------------------------')
        if curr_vid_label > 0:
            img_file = os.path.join(self.img_dataset_path, "positive",
                                    feature_path.split('/')[-1].split(".")[0] + '.npy')
            # logging.info('feature_path positive: ', feature_path.split('/'))
        else:
            img_file = os.path.join(self.img_dataset_path, "negative",
                                    feature_path.split('/')[-1].split(".")[0] + '.npy')
            # logging.info('feature_path negative: ', feature_path.split('/'))
        all_img_feat = self.transform(np.load(img_file)).squeeze(0)

        # Reading frame stats file
        if curr_vid_label > 0:
            frame_stats_file = os.path.join(self.frame_stats_path, "positive",
                                            feature_path.split('/')[-1].split(".")[0] + '.npy')
            # logging.info('feature_path positive: ', feature_path.split('/'))
        else:
            frame_stats_file = os.path.join(self.frame_stats_path, "negative",
                                            feature_path.split('/')[-1].split(".")[0] + '.npy')
            # logging.info('feature_path negative: ', feature_path.split('/'))
        frame_stats = torch.from_numpy(np.load(frame_stats_file)).float()

        # Calculating the bbox centers
        cx, cy = (all_bbox[:, :, 0] + all_bbox[:, :, 2]) / 2, (all_bbox[:, :, 1] + all_bbox[:, :, 3]) / 2
        all_obj_centers = torch.cat((cx.unsqueeze(2), cy.unsqueeze(2)), 2).float()

        # Obj label indexes
        all_obj_label_idxs = all_bbox[:, :, -1].long()

        # ---------- Building the object-level spatio-temporal graph ---------------
        video_data, edge_weights, edge_embeddings, obj_vis_feat = [], [], [], []
        start_idx = 0
        actual_frame_idx = 0
        obj_refs = {}
        num_objs_list = []

        for i in range(all_feat.shape[0]):

            obj_label_idxs, obj_feat, obj_centers, bbox = all_obj_label_idxs[i], all_feat[i], all_obj_centers[i], all_bbox[i]
            num_objs = len(obj_label_idxs)

            # print()
            # print(obj_label_idxs.shape, obj_feat.shape, obj_centers.shape)

            # Create the adj list (making big disconnected graph for one video)
            # relation pair idxs
            source_nodes = torch.arange(obj_label_idxs.shape[0]).unsqueeze(1).repeat(1,
                                                                                     obj_label_idxs.shape[0]).flatten()
            target_nodes = torch.arange(obj_label_idxs.shape[0]).unsqueeze(0).repeat(1,
                                                                                     obj_label_idxs.shape[0]).flatten()

            # print(source_nodes, target_nodes)
            
            adj_list = torch.cat((source_nodes.unsqueeze(1), target_nodes.unsqueeze(1)), 1)
            repeat_idx = torch.where(adj_list[:, 0] != adj_list[:, 1])[0]  # removing the self loops from the adj list
            adj_list = adj_list[repeat_idx]
            adj_list = torch.LongTensor(adj_list)
            adj_list = adj_list.permute((1, 0))

            # print("adj_list ", adj_list.shape)

            # Edge embeddings - dist_rel, obj centers of source, obj centers of target
            source_nodes, target_nodes = adj_list[0, :], adj_list[1, :]
            obj_centers = obj_centers / frame_stats[i] 
            bbox[:, 0:2] = bbox[:, 0:2] / frame_stats[i]  # normalize with frame height and width
            bbox[:, 2:4] = bbox[:, 2:4] / frame_stats[i]
            source_centers = obj_centers[source_nodes]
            target_centers = obj_centers[target_nodes]

            # print(source_nodes, target_nodes)

            # print(source_centers.shape, target_centers.shape)

            # dist_mat = self.get_distance(source_centers, target_centers).float()
            dist_mat = torch.cdist(source_centers, target_centers).float()

            
            # print("dist_mat ", dist_mat1.shape, dist_mat2.shape)
            dist_mat = F.softmax(-dist_mat, dim=-1)
            dist_rel = dist_mat[adj_list[0, :], adj_list[1, :]].unsqueeze(1)

            if self.ego_dist == 1:
                n = bbox.shape[0]-1
                dist = self._get_distances((bbox[0, 0].unsqueeze(0).repeat(n, 1), bbox[0, 1].unsqueeze(0).repeat(n, 1)), (bbox[0, 2].unsqueeze(0).repeat(n, 1), bbox[0, 3].unsqueeze(0).repeat(n, 1)), ((bbox[1:, 0]+bbox[1:, 2])/2, bbox[1:, 3]))
                dist = torch.nan_to_num(dist, nan=0.0)
                dist_rel[0:n, :] = dist[0].unsqueeze(1)

            edge_embed = torch.cat((source_centers, target_centers, dist_rel), 1)

            adj_list += start_idx

            target = curr_vid_label  # Target

            # Adding obj label embeddings to the node features
            label_embeddings = self.obj_embeddings[obj_label_idxs]
            x_embed = torch.cat((obj_feat, label_embeddings), 1)
            frame_data = Data(x=x_embed, edge_index=adj_list, y=target)

            video_data.append(frame_data)
            edge_embeddings.append(edge_embed)
            obj_vis_feat.append(obj_feat)

            obj_refs['f_' + str(actual_frame_idx)] = {}
            unique_obj_idxs = torch.unique(obj_label_idxs)
            for obj_ in unique_obj_idxs:
                idxx = torch.where(obj_label_idxs == obj_)[0] + start_idx
                obj_refs['f_' + str(actual_frame_idx)][obj_.item()] = idxx

            start_idx += num_objs

            # keep track of num of objects for pooling to create graph-embedding later
            num_objs_list += [torch.zeros(num_objs) + actual_frame_idx]
            actual_frame_idx += 1

        data, slices = InMemoryDataset.collate(video_data)
        edge_embeddings = torch.cat(edge_embeddings)
        obj_vis_feat = torch.cat(obj_vis_feat)
        num_objs_list = torch.cat(num_objs_list)

        # Generate the temporal connections adjacency matrix for the object-level graph
        temporal_adj_list, temporal_edge_w = [], []
        for i, (key_, value_) in enumerate(obj_refs.items()):
            temporal_ref = min(i, self.temporal_ref)
            curr_frame = obj_refs['f_' + str(i)]
            for t_ in range(1, temporal_ref + 1):
                prev_frame = obj_refs['f_' + str(i - t_)]
                for obj_l, t_n in curr_frame.items():
                    if obj_l in prev_frame:
                        s_n = prev_frame[obj_l]  # source node idxs
                        num_s = s_n.shape[0]
                        # repeat all sources nodes length of target times - for example if source = [1, 2] and len(target)=2 it becomes [1,1,2,2]
                        s_n = s_n.unsqueeze(0).repeat(len(t_n), 1).T.reshape(-1, 1)
                        t_n = t_n.repeat(num_s).unsqueeze(1)  # repeat target source times directly
                        s_t = torch.cat((s_n, t_n), 1)
                        # temporal_edge_w.append(feat_sim[s_t[:, 0], s_t[:, 1]])    #choose the feature similarity as edge weight
                        temporal_adj_list += s_t.tolist()

        temporal_adj_list = torch.Tensor(temporal_adj_list).permute((1, 0)).long()

        # Generate the frame-level adjacency matrix
        video_adj_list = []
        for i in range(len(video_data)):
            bw_ref_interval = min(i, self.ref_interval)
            for j in range(1, bw_ref_interval + 1, self.dilation_factor):
                video_adj_list += [[i - j, i]]  # adding previous ref_interval neighbors
        video_adj_list = torch.Tensor(video_adj_list).permute((1, 0)).long()

        return data.x, data.edge_index, data.y, all_img_feat, video_adj_list, edge_embeddings, temporal_adj_list, obj_vis_feat, num_objs_list, curr_toa

    def __len__(self):
        return len(self.feature_paths)

class FeaturesDataset(Dataset):
    def __init__(self, dataset_path, img_dataset_path, toas_files_path, split_path, ref_interval, objmap_file, training):

        """
        Input:
        dataset_path: Path to the video frames
        img_dataset_path: Path to the I3D frame features
        split_path: Path to the split files
        ref_interval: Number of frames temporally connected in frame-level graph
        objmap_file: The file mapping labels to label names for objects
        annofile: File containing TOA annotations
        training: Flag to determine train/test phase

        """

        self.training = training
        self.img_dataset_path = img_dataset_path
        self.feature_paths = self._extract_feature_paths(dataset_path, split_path, training)
        self.transform = transforms.Compose([transforms.ToTensor(), ])
        # self.frame_batch_size = frame_batch_size
        self.ref_interval = ref_interval
        self.temporal_ref = 1
        self.dilation_factor = 1
        self.toas_files_path = toas_files_path
        self.topk = 10
        self.frame_stats_path = dataset_path[:-8] + 'frames_stats'  # (height, width)
        self.n_frames = 50

        # Obj label to word embeddings
        self.idx_to_classes_obj = json.load(open(objmap_file))
        self.nlp = spacy.load('en_core_web_md', disable=['ner', 'parser'])
        self.obj_embeddings = torch.from_numpy(np.array([self.nlp(obj).vector for obj in self.idx_to_classes_obj]))

    def _extract_feature_paths(self, dataset_path, split_path="splits_dota/", training=True):

        """ Function to extract paths to frames given the specified train/test split
        Input:
        dataset_path: Path to the video frames
        split_path: Path to the split files
        training: Flag to determine train/test phase

        Returns:
        feature_paths: List of all the video paths in a split

        """
        fn = "split.txt"
        split_path = os.path.join(split_path, fn)
        with open(split_path) as file:
            lines = file.read().splitlines()
        feature_paths = []

        for line in lines:
            feature_paths += [os.path.join(dataset_path, line)]

        return feature_paths

    def _frame_number(self, feat_path):

        """ Function to extract the frame number from the input
        Input:
        feat_path: Input path for a frame

        Returns:
        Integer of the frame number (temporal position of a frame in a video)

        """
        return int(feat_path.split('/')[-1].split('.mat')[0].split('_')[-1])

    def _get_distances(self, a, b, p):
        return torch.abs(((b[1] - a[1])*p[0] - (b[0] - a[0])*p[1] + b[0]*a[1] - b[1]*a[0]) / torch.sqrt(torch.pow(b[1] - a[1], 2) + torch.pow(b[0] - a[0], 2)))

    def get_toa_all(self, video_name):

        """ 
        get toa from toas directory
        """

        toa_dir = self.toas_files_path
        toa_file = os.path.join(toa_dir, video_name+'.txt')

        with open(toa_file) as f:
            return int(f.read())
        

    def __getitem__(self, index):

        feature_path = self.feature_paths[index % (len(self))]
        video_name = feature_path.split('/')[-1][:-4]

        # Load the data.npy (for features) and det.npy (for bounding boxes) files
        all_data = np.load(f"{feature_path}")
        all_feat = torch.from_numpy(all_data['data'])[:, 1:, :]
        all_bbox = torch.from_numpy(
            all_data['det']).float()  # (x1, y1, x2, y2, cls, accident/no acc)bottom left and top right coordinates

        curr_vid_label = int(all_data['labels'][1])
        if curr_vid_label > 0:
            curr_toa = self.get_toa_all(video_name)
        else:
            curr_toa = self.n_frames + 1

        # Reading frame (i3d) features for the frames
        if curr_vid_label > 0:
            img_file = os.path.join(self.img_dataset_path, "positive",
                                    feature_path.split('/')[-1].split(".")[0] + '.npy')
        else:
            img_file = os.path.join(self.img_dataset_path, "negative",
                                    feature_path.split('/')[-1].split(".")[0] + '.npy')
        all_img_feat = self.transform(np.load(img_file)).squeeze(0)

        # Reading frame stats file
        if curr_vid_label > 0:
            frame_stats_file = os.path.join(self.frame_stats_path, "positive",
                                            feature_path.split('/')[-1].split(".")[0] + '.npy')
        else:
            frame_stats_file = os.path.join(self.frame_stats_path, "negative",
                                            feature_path.split('/')[-1].split(".")[0] + '.npy')
        frame_stats = torch.from_numpy(np.load(frame_stats_file)).float()

        # Calculating the bbox centers
        cx, cy = (all_bbox[:, :, 0] + all_bbox[:, :, 2]) / 2, (all_bbox[:, :, 1] + all_bbox[:, :, 3]) / 2
        all_obj_centers = torch.cat((cx.unsqueeze(2), cy.unsqueeze(2)), 2).float()

        # Obj label indexes
        all_obj_label_idxs = all_bbox[:, :, -1].long()

        # ---------- Building the object-level spatio-temporal graph ---------------
        video_data, edge_weights, edge_embeddings, obj_vis_feat = [], [], [], []
        start_idx = 0
        actual_frame_idx = 0
        obj_refs = {}
        num_objs_list = []

        for i in range(all_feat.shape[0]):

            obj_label_idxs, obj_feat, obj_centers, bbox = all_obj_label_idxs[i], all_feat[i], all_obj_centers[i], all_bbox[i]
            num_objs = len(obj_label_idxs)

            # print()
            # print(obj_label_idxs.shape, obj_feat.shape, obj_centers.shape)

            # Create the adj list (making big disconnected graph for one video)
            # relation pair idxs
            source_nodes = torch.arange(obj_label_idxs.shape[0]).unsqueeze(1).repeat(1,
                                                                                     obj_label_idxs.shape[0]).flatten()
            target_nodes = torch.arange(obj_label_idxs.shape[0]).unsqueeze(0).repeat(1,
                                                                                     obj_label_idxs.shape[0]).flatten()

            # print(source_nodes, target_nodes)
            
            adj_list = torch.cat((source_nodes.unsqueeze(1), target_nodes.unsqueeze(1)), 1)
            repeat_idx = torch.where(adj_list[:, 0] != adj_list[:, 1])[0]  # removing the self loops from the adj list
            adj_list = adj_list[repeat_idx]
            adj_list = torch.LongTensor(adj_list)
            adj_list = adj_list.permute((1, 0))

            # print("adj_list ", adj_list.shape)

            # Edge embeddings - dist_rel, obj centers of source, obj centers of target
            source_nodes, target_nodes = adj_list[0, :], adj_list[1, :]
            obj_centers = obj_centers / frame_stats[i] 
            bbox[:, 0:2] = bbox[:, 0:2] / frame_stats[i]  # normalize with frame height and width
            bbox[:, 2:4] = bbox[:, 2:4] / frame_stats[i]
            source_centers = obj_centers[source_nodes]
            target_centers = obj_centers[target_nodes]

            # print(source_nodes, target_nodes)

            # print(source_centers.shape, target_centers.shape)

            # dist_mat = self.get_distance(source_centers, target_centers).float()
            dist_mat = torch.cdist(source_centers, target_centers).float()

            n = bbox.shape[0]-1

            dist = self._get_distances((bbox[0, 0].unsqueeze(0).repeat(n, 1), bbox[0, 1].unsqueeze(0).repeat(n, 1)), (bbox[0, 2].unsqueeze(0).repeat(n, 1), bbox[0, 3].unsqueeze(0).repeat(n, 1)), ((bbox[1:, 0]+bbox[1:, 2])/2, bbox[1:, 3]))
            dist = torch.nan_to_num(dist, nan=0.0)
            
            # print("dist_mat ", dist_mat1.shape, dist_mat2.shape)
            dist_mat = F.softmax(-dist_mat, dim=-1)
            dist_rel = dist_mat[adj_list[0, :], adj_list[1, :]].unsqueeze(1)

            dist_rel[0:n, :] = dist[0].unsqueeze(1)

            edge_embed = torch.cat((source_centers, target_centers, dist_rel), 1)

            adj_list += start_idx

            target = curr_vid_label  # Target

            # Adding obj label embeddings to the node features
            label_embeddings = self.obj_embeddings[obj_label_idxs]
            x_embed = torch.cat((obj_feat, label_embeddings), 1)
            frame_data = Data(x=x_embed, edge_index=adj_list, y=target)

            video_data.append(frame_data)
            edge_embeddings.append(edge_embed)
            obj_vis_feat.append(obj_feat)

            obj_refs['f_' + str(actual_frame_idx)] = {}
            unique_obj_idxs = torch.unique(obj_label_idxs)
            for obj_ in unique_obj_idxs:
                idxx = torch.where(obj_label_idxs == obj_)[0] + start_idx
                obj_refs['f_' + str(actual_frame_idx)][obj_.item()] = idxx

            start_idx += num_objs

            # keep track of num of objects for pooling to create graph-embedding later
            num_objs_list += [torch.zeros(num_objs) + actual_frame_idx]
            actual_frame_idx += 1

        data, slices = InMemoryDataset.collate(video_data)
        edge_embeddings = torch.cat(edge_embeddings)
        obj_vis_feat = torch.cat(obj_vis_feat)
        num_objs_list = torch.cat(num_objs_list)

        # Generate the temporal connections adjacency matrix for the object-level graph
        temporal_adj_list, temporal_edge_w = [], []
        for i, (key_, value_) in enumerate(obj_refs.items()):
            temporal_ref = min(i, self.temporal_ref)
            curr_frame = obj_refs['f_' + str(i)]
            for t_ in range(1, temporal_ref + 1):
                prev_frame = obj_refs['f_' + str(i - t_)]
                for obj_l, t_n in curr_frame.items():
                    if obj_l in prev_frame:
                        s_n = prev_frame[obj_l]  # source node idxs
                        num_s = s_n.shape[0]
                        # repeat all sources nodes length of target times - for example if source = [1, 2] and len(target)=2 it becomes [1,1,2,2]
                        s_n = s_n.unsqueeze(0).repeat(len(t_n), 1).T.reshape(-1, 1)
                        t_n = t_n.repeat(num_s).unsqueeze(1)  # repeat target source times directly
                        s_t = torch.cat((s_n, t_n), 1)
                        # temporal_edge_w.append(feat_sim[s_t[:, 0], s_t[:, 1]])    #choose the feature similarity as edge weight
                        temporal_adj_list += s_t.tolist()

        temporal_adj_list = torch.Tensor(temporal_adj_list).permute((1, 0)).long()

        # Generate the frame-level adjacency matrix
        video_adj_list = []
        for i in range(len(video_data)):
            bw_ref_interval = min(i, self.ref_interval)
            for j in range(1, bw_ref_interval + 1, self.dilation_factor):
                video_adj_list += [[i - j, i]]  # adding previous ref_interval neighbors
        video_adj_list = torch.Tensor(video_adj_list).permute((1, 0)).long()

        return data.x, data.edge_index, data.y, all_img_feat, video_adj_list, edge_embeddings, temporal_adj_list, obj_vis_feat, num_objs_list, curr_toa, feature_path.split('/')[-1].split(".")[0]

    def __len__(self):
        return len(self.feature_paths)
