import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, global_max_pool, GATv2Conv, TopKPooling, SAGPooling
from torch_geometric.nn.norm import InstanceNorm
import copy
import sys

# ------------------- Model for accident prevention/detection (CCD dataset) task--------------------
class SpaceTempGoG_detr_ccd(nn.Module):

	def __init__(self, input_dim=4096, embedding_dim=128, img_feat_dim=2048, num_classes=2):
		super(SpaceTempGoG_detr_ccd, self).__init__()

		self.num_heads = 1
		self.input_dim = input_dim

		#process the object graph features 
		self.x_fc = nn.Linear(self.input_dim, embedding_dim*2)
		self.x_bn1 = nn.BatchNorm1d(embedding_dim*2)
		self.obj_l_fc = nn.Linear(300, embedding_dim//2)
		self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim//2)

		#GNN for encoding the object-level graph
		self.gc1_spatial = GCNConv(embedding_dim*2+embedding_dim//2, embedding_dim//2)   
		self.gc1_norm1 = InstanceNorm(embedding_dim//2)
		self.gc1_temporal = GCNConv(embedding_dim*2+embedding_dim//2, embedding_dim//2)  
		self.gc1_norm2 = InstanceNorm(embedding_dim//2)
		self.pool = TopKPooling(embedding_dim, ratio=0.8)

		#I3D features processing
		self.img_fc = nn.Linear(img_feat_dim, embedding_dim*2)         

		# Frame-level graph
		self.gc2_sg = GATv2Conv(embedding_dim, embedding_dim//2, heads=self.num_heads)  #+
		self.gc2_norm1 = InstanceNorm((embedding_dim//2)*self.num_heads)
		self.gc2_i3d = GATv2Conv(embedding_dim*2, embedding_dim//2, heads=self.num_heads)
		self.gc2_norm2 = InstanceNorm((embedding_dim//2)*self.num_heads)

		self.classify_fc1 = nn.Linear(embedding_dim, embedding_dim//2)
		self.classify_fc2 = nn.Linear(embedding_dim//2, num_classes)

		self.relu = nn.LeakyReLU(0.2)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):

		"""
		Inputs: 
		x - object-level graph nodes' feature matrix 
		edge_index - spatial graph connectivity for object-level graph 
		img_feat - frame I3D features 
		video_adj_list - Graph connectivity for frame-level graph
		edge_embeddings - Edge features for the object-level graph
		temporal_adj_list - temporal graph connectivity for object-level graph 
		temporal_wdge_w - edge weights for frame-level graph 
		batch_vec - vector for graph pooling the object-level graph
		
		Returns: 
		logits_mc - Final logits 
		probs_mc - Final probabilities
		"""
		
		#process graph inputs 
		x_feat = self.x_fc(x[:, :self.input_dim])
		x_feat = self.relu(self.x_bn1(x_feat))
		x_label = self.obj_l_fc(x[:, self.input_dim:])
		x_label = self.relu(self.obj_l_bn1(x_label))
		x = torch.cat((x_feat, x_label), 1)
        
		#Get graph embedding for object-level graph
		n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_weight=edge_embeddings[:, -1])))
		n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, temporal_edge_w)))
		n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
		n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
		g_embed = global_max_pool(n_embed, batch_vec)

		# Process I3D feature
		img_feat = self.img_fc(img_feat)
		
		# Get frame embedding for all nodes in frame-level graph
		frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
		frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list)))
		frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img), 1)
		frame_embed_sg = self.relu(self.classify_fc1(frame_embed_))
		logits_mc = self.classify_fc2(frame_embed_sg)
		probs_mc = self.softmax(logits_mc)
		
		return logits_mc, probs_mc

# ------------------- Model for accident prevention/detection (DAD dataset) task--------------------
# class SpaceTempGoG_detr_dad(nn.Module):

# 	def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
# 		super(SpaceTempGoG_detr_dad, self).__init__()

# 		self.num_heads = 1
# 		self.input_dim = input_dim

# 		#process the object graph features 
# 		self.x_fc = nn.Linear(self.input_dim, embedding_dim*2)
# 		self.x_bn1 = nn.BatchNorm1d(embedding_dim*2)
# 		self.obj_l_fc = nn.Linear(300, embedding_dim//2)
# 		self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim//2)

# 		# GNN for encoding the object-level graph 
# 		self.gc1_spatial = GCNConv(embedding_dim*2+embedding_dim//2, embedding_dim//2)   
# 		self.gc1_norm1 = InstanceNorm(embedding_dim//2)
# 		self.gc1_temporal = GCNConv(embedding_dim*2+embedding_dim//2, embedding_dim//2)   
# 		self.gc1_norm2 = InstanceNorm(embedding_dim//2)
# 		self.pool = TopKPooling(embedding_dim, ratio=0.8)

# 		#I3D features
# 		self.img_fc = nn.Linear(img_feat_dim, embedding_dim*2)

# 		self.gc2_sg = GATv2Conv(embedding_dim, embedding_dim//2, heads=self.num_heads)  #+
# 		self.gc2_norm1 = InstanceNorm((embedding_dim//2)*self.num_heads)
# 		self.gc2_i3d = GATv2Conv(embedding_dim*2, embedding_dim//2, heads=self.num_heads)
# 		self.gc2_norm2 = InstanceNorm((embedding_dim//2)*self.num_heads)

# 		self.classify_fc1 = nn.Linear(embedding_dim, embedding_dim//2)
# 		self.classify_fc2 = nn.Linear(embedding_dim//2, num_classes)

# 		self.relu = nn.LeakyReLU(0.2)
# 		self.softmax = nn.Softmax(dim=-1)

# 	def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):

# 		"""
# 		Inputs: 
# 		x - object-level graph nodes' feature matrix 
# 		edge_index - spatial graph connectivity for object-level graph 
# 		img_feat - frame I3D features 
# 		video_adj_list - Graph connectivity for frame-level graph
# 		edge_embeddings - Edge features for the object-level graph
# 		temporal_adj_list - temporal graph connectivity for object-level graph 
# 		temporal_wdge_w - edge weights for frame-level graph 
# 		batch_vec - vector for graph pooling the object-level graph
		
# 		Returns: 
# 		logits_mc - Final logits 
# 		probs_mc - Final probabilities
# 		"""
		
# 		#process object graph features 
# 		x_feat = self.x_fc(x[:, :self.input_dim])
# 		x_feat = self.relu(self.x_bn1(x_feat))
# 		x_label = self.obj_l_fc(x[:, self.input_dim:])
# 		x_label = self.relu(self.obj_l_bn1(x_label))
# 		x = torch.cat((x_feat, x_label), 1)
        
# 		#Get graph embedding for ibject-level graph
# 		n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_weight=edge_embeddings[:, -1])))
# 		n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, temporal_edge_w)))
# 		n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
# 		n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
# 		g_embed = global_max_pool(n_embed, batch_vec)

# 		#Process I3D feature
# 		img_feat = self.img_fc(img_feat)

# 		#Get frame embedding for all nodes in frame-level graph
# 		frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
# 		frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list)))
# 		frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img), 1)
# 		frame_embed_sg = self.relu(self.classify_fc1(frame_embed_))
# 		logits_mc = self.classify_fc2(frame_embed_sg)
# 		probs_mc = self.softmax(logits_mc)
		
# 		return logits_mc, probs_mc 

class SpaceTempGoG_detr_dad(nn.Module):

	def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
		super(SpaceTempGoG_detr_dad, self).__init__()

		self.num_heads = 1
		self.input_dim = input_dim

		#process the object graph features 
		self.x_fc = nn.Linear(self.input_dim, embedding_dim*2)
		self.x_bn1 = nn.BatchNorm1d(embedding_dim*2)
		self.obj_l_fc = nn.Linear(300, embedding_dim//2)
		self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim//2)

		# GNN for encoding the object-level graph 
		# self.gc1_spatial = GCNConv(embedding_dim*2+embedding_dim//2, embedding_dim//2)  
		
	        # Improved GNN for encoding the object-level graph
		self.gc1_spatial = GATv2Conv(
		embedding_dim * 2 + embedding_dim // 2, 
		embedding_dim // 2, 
		heads=self.num_heads,
		edge_dim=1  # Using temporal_edge_w as edge features
		) 
		self.gc1_norm1 = InstanceNorm(embedding_dim//2)

		# self.gc1_temporal = GCNConv(embedding_dim*2+embedding_dim//2, embedding_dim//2)   
		
        	# Improved temporal graph convolution
		self.gc1_temporal = GATv2Conv(
		embedding_dim * 2 + embedding_dim // 2, 
		embedding_dim // 2, 
		heads=self.num_heads,
		edge_dim=1  # Using temporal_edge_w as edge features
		)

		self.gc1_norm2 = InstanceNorm(embedding_dim//2)
		# self.pool = TopKPooling(embedding_dim, ratio=0.8)
		self.pool = SAGPooling(embedding_dim, ratio=0.8)

		#I3D features
		self.img_fc = nn.Linear(img_feat_dim, embedding_dim*2)

		# # Added LSTM for temporal sequence processing
		self.temporal_lstm = nn.LSTM(
		input_size=embedding_dim * 2,
		hidden_size=embedding_dim * 2,  # Changed to match input size
		num_layers=1,
		batch_first=True
		)

		self.gc2_sg = GATv2Conv(embedding_dim, embedding_dim//2, heads=self.num_heads)  #+
		self.gc2_norm1 = InstanceNorm((embedding_dim//2)*self.num_heads)
		self.gc2_i3d = GATv2Conv(embedding_dim*2, embedding_dim//2, heads=self.num_heads)
		self.gc2_norm2 = InstanceNorm((embedding_dim//2)*self.num_heads)

		self.classify_fc1 = nn.Linear(embedding_dim, embedding_dim//2)
		self.classify_fc2 = nn.Linear(embedding_dim//2, num_classes)

		self.relu = nn.LeakyReLU(0.2)
		self.softmax = nn.Softmax(dim=-1)

	def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):

		"""
		Inputs: 
		x - object-level graph nodes' feature matrix 
		edge_index - spatial graph connectivity for object-level graph 
		img_feat - frame I3D features 
		video_adj_list - Graph connectivity for frame-level graph
		edge_embeddings - Edge features for the object-level graph
		temporal_adj_list - temporal graph connectivity for object-level graph 
		temporal_wdge_w - edge weights for frame-level graph 
		batch_vec - vector for graph pooling the object-level graph
		
		Returns: 
		logits_mc - Final logits 
		probs_mc - Final probabilities
		"""
		
		#process object graph features 
		x_feat = self.x_fc(x[:, :self.input_dim])
		x_feat = self.relu(self.x_bn1(x_feat))
		x_label = self.obj_l_fc(x[:, self.input_dim:])
		x_label = self.relu(self.obj_l_bn1(x_label))
		x = torch.cat((x_feat, x_label), 1)
        
		#Get graph embedding for ibject-level graph
		# n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_weight=edge_embeddings[:, -1])))
	        
		# Improved Get graph embedding for object-level graph
		n_embed_spatial = self.relu(self.gc1_norm1(
		self.gc1_spatial(x, edge_index, edge_attr=edge_embeddings[:, -1].unsqueeze(1))
		))
		
		#old
		# n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, temporal_edge_w)))
		
		# Improved temporal processing
		temporal_edge_w = temporal_edge_w.to(x.dtype)
		n_embed_temporal = self.relu(self.gc1_norm2(
		self.gc1_temporal(x, temporal_adj_list, edge_attr=temporal_edge_w.unsqueeze(1))
		))
		
		n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
		n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
		g_embed = global_max_pool(n_embed, batch_vec)
		
		#Process I3D feature
		img_feat = self.img_fc(img_feat)
		
		# change - LSTM processing - reshape for temporal dimension
		img_feat = img_feat.unsqueeze(0)  # Add sequence dimension (1, num_nodes, features)
		img_feat, (_, _) = self.temporal_lstm(img_feat)  # Extract only output, discard hidden and cell state
		img_feat = img_feat.squeeze(0)  # Back to (num_nodes, features)	
		
		#Get frame embedding for all nodes in frame-level graph
		frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
		frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list)))
		frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img), 1)
		frame_embed_sg = self.relu(self.classify_fc1(frame_embed_))
		logits_mc = self.classify_fc2(frame_embed_sg)
		probs_mc = self.softmax(logits_mc)
		
		return logits_mc, probs_mc
		

# class SpaceTempGoG_detr_dota(nn.Module):

# 	def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
# 		super(SpaceTempGoG_detr_dota, self).__init__()

# 		self.num_heads = 1
# 		self.input_dim = input_dim

# 		# process the object graph features
# 		self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
# 		self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
# 		self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
# 		self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

# 		# GNN for encoding the object-level graph
# 		self.gc1_spatial = GCNConv(embedding_dim * 2 + embedding_dim // 2, embedding_dim // 2)
# 		self.gc1_norm1 = InstanceNorm(embedding_dim // 2)
# 		self.gc1_temporal = GCNConv(embedding_dim * 2 + embedding_dim // 2, embedding_dim // 2)
# 		self.gc1_norm2 = InstanceNorm(embedding_dim // 2)
# 		self.pool = TopKPooling(embedding_dim, ratio=0.8)

# 		# I3D features
# 		self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

# 		self.gc2_sg = GATv2Conv(embedding_dim, embedding_dim // 2, heads=self.num_heads)  # +
# 		self.gc2_norm1 = InstanceNorm((embedding_dim // 2) * self.num_heads)
# 		self.gc2_i3d = GATv2Conv(embedding_dim * 2, embedding_dim // 2, heads=self.num_heads)
# 		self.gc2_norm2 = InstanceNorm((embedding_dim // 2) * self.num_heads)

# 		self.classify_fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
# 		self.classify_fc2 = nn.Linear(embedding_dim // 2, num_classes)

# 		self.relu = nn.LeakyReLU(0.2)
# 		self.softmax = nn.Softmax(dim=-1)

# 	def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, temporal_edge_w,
# 				batch_vec):
# 		"""
# 		Inputs:
# 		x - object-level graph nodes' feature matrix
# 		edge_index - spatial graph connectivity for object-level graph
# 		img_feat - frame I3D features
# 		video_adj_list - Graph connectivity for frame-level graph
# 		edge_embeddings - Edge features for the object-level graph
# 		temporal_adj_list - temporal graph connectivity for object-level graph
# 		temporal_wdge_w - edge weights for frame-level graph
# 		batch_vec - vector for graph pooling the object-level graph

# 		Returns:
# 		logits_mc - Final logits
# 		probs_mc - Final probabilities
# 		"""

# 		# process object graph features
# 		x_feat = self.x_fc(x[:, :self.input_dim])
# 		x_feat = self.relu(self.x_bn1(x_feat))
# 		x_label = self.obj_l_fc(x[:, self.input_dim:])
# 		x_label = self.relu(self.obj_l_bn1(x_label))
# 		x = torch.cat((x_feat, x_label), 1)

# 		# Get graph embedding for ibject-level graph
# 		n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_weight=edge_embeddings[:, -1])))
# 		n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, temporal_edge_w)))
# 		n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
# 		n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
# 		g_embed = global_max_pool(n_embed, batch_vec)

# 		# Process I3D feature
# 		img_feat = self.img_fc(img_feat)

# 		# Get frame embedding for all nodes in frame-level graph
# 		frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
# 		frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list)))
# 		frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img), 1)
# 		frame_embed_sg = self.relu(self.classify_fc1(frame_embed_))
# 		logits_mc = self.classify_fc2(frame_embed_sg)
# 		probs_mc = self.softmax(logits_mc)

# 		return logits_mc, probs_mc

from torch_geometric.nn import (
    GATv2Conv, 
    TopKPooling,
    SAGPooling,
    global_max_pool, 
    global_mean_pool,
    InstanceNorm
)
from torch.nn import Sequential, Linear, BatchNorm1d, LeakyReLU, Dropout, GRU, MultiheadAttention

class SpaceTempGoG_detr_dota(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(SpaceTempGoG_detr_dota, self).__init__()

        self.num_heads = 1
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # process the object graph features
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)
        
        # Improved GNN for encoding the object-level graph
        self.gc1_spatial = GATv2Conv(
            embedding_dim * 2 + embedding_dim // 2, 
            embedding_dim // 2, 
            heads=self.num_heads,
            edge_dim=1  # Using temporal_edge_w as edge features
        )
        # GNN for encoding the object-level graph
        # self.gc1_spatial = GCNConv(embedding_dim * 2 + embedding_dim // 2, embedding_dim // 2)
        self.gc1_norm1 = InstanceNorm(embedding_dim // 2)
        
        # Improved temporal graph convolution
        self.gc1_temporal = GATv2Conv(
            embedding_dim * 2 + embedding_dim // 2, 
            embedding_dim // 2, 
            heads=self.num_heads,
            edge_dim=1  # Using temporal_edge_w as edge features
        )
        # self.gc1_temporal = GCNConv(embedding_dim * 2 + embedding_dim // 2, embedding_dim // 2)
        self.gc1_norm2 = InstanceNorm(embedding_dim // 2)  # Removed *num_heads since we're using 1 head
        
        self.pool = TopKPooling(embedding_dim, ratio=0.8)
        # self.pool = SAGPooling(embedding_dim, ratio=0.8)

        # I3D features with temporal processing
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
        
        # # Added GRU for temporal sequence processing
        # self.temporal_gru = nn.GRU(
        #     input_size=embedding_dim * 2,
        #     hidden_size=embedding_dim * 2,  # Changed to match input size
        #     num_layers=1,
        #     batch_first=True
        # )

        # Added LSTM for temporal sequence processing
        # self.temporal_lstm = nn.LSTM(
        #     input_size=embedding_dim * 2,
        #     hidden_size=embedding_dim * 2,  # Changed to match input size
        #     num_layers=1,
        #     batch_first=True
        # )

        # Fixed dimension mismatches in these layers
        self.gc2_sg = GATv2Conv(
            embedding_dim,  # Input from g_embed
            embedding_dim // 2, 
            heads=self.num_heads
        )
        self.gc2_norm1 = InstanceNorm(embedding_dim // 2)
        
        self.gc2_i3d = GATv2Conv(
            embedding_dim * 2,  # Input from GRU output
            embedding_dim // 2, 
            heads=self.num_heads
        )
        self.gc2_norm2 = InstanceNorm(embedding_dim // 2)

        self.classify_fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
        self.classify_fc2 = nn.Linear(embedding_dim // 2, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):
        # process object graph features
        x_feat = self.x_fc(x[:, :self.input_dim])
        x_feat = self.relu(self.x_bn1(x_feat))
        x_label = self.obj_l_fc(x[:, self.input_dim:])
        x_label = self.relu(self.obj_l_bn1(x_label))
        x = torch.cat((x_feat, x_label), 1)

        # Old Get graph embedding for object-level graph
        # n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_weight=edge_embeddings[:, -1])))
        
        # Improved Get graph embedding for object-level graph
        n_embed_spatial = self.relu(self.gc1_norm1(
            self.gc1_spatial(x, edge_index, edge_attr=edge_embeddings[:, -1].unsqueeze(1))
        ))
        
        # Old temporal processing
        # n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, temporal_edge_w)))
        
        # Improved temporal processing
        n_embed_temporal = self.relu(self.gc1_norm2(
            self.gc1_temporal(x, temporal_adj_list, edge_attr=temporal_edge_w.unsqueeze(1))
        ))
        
        n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
        g_embed = global_max_pool(n_embed, batch_vec)

        # Process I3D feature with temporal modeling
        img_feat = self.img_fc(img_feat)
        # print("After img_fc:", img_feat.shape)
        
        # GRU processing - reshape for temporal dimension
        # img_feat = img_feat.unsqueeze(0)  # Add sequence dimension (1, num_nodes, features)
        # img_feat, _ = self.temporal_gru(img_feat)
        # img_feat = img_feat.squeeze(0)  # Back to (num_nodes, features)

	# # LSTM processing - reshape for temporal dimension
 #        img_feat = img_feat.unsqueeze(0)  # Add sequence dimension (1, num_nodes, features)
 #        # print("After unsqueeze:", img_feat.shape)
 #        img_feat, (_, _) = self.temporal_lstm(img_feat)  # Extract only output, discard hidden and cell state
 #        # print("After temporal_lstm:", img_feat.shape)
 #        img_feat = img_feat.squeeze(0)  # Back to (num_nodes, features)
 #        # print("After squeeze:", img_feat.shape)

        # Get frame embedding for all nodes in frame-level graph
        frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
        frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list)))
        frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img), 1)
        frame_embed_sg = self.relu(self.classify_fc1(frame_embed_))
        logits_mc = self.classify_fc2(frame_embed_sg)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc
