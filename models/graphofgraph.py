import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, global_max_pool, GATv2Conv, TopKPooling
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
		self.gc1_spatial = GCNConv(embedding_dim*2+embedding_dim//2, embedding_dim//2)   
		self.gc1_norm1 = InstanceNorm(embedding_dim//2)
		self.gc1_temporal = GCNConv(embedding_dim*2+embedding_dim//2, embedding_dim//2)   
		self.gc1_norm2 = InstanceNorm(embedding_dim//2)
		self.pool = TopKPooling(embedding_dim, ratio=0.8)

		#I3D features
		self.img_fc = nn.Linear(img_feat_dim, embedding_dim*2)         

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
		n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_weight=edge_embeddings[:, -1])))
		n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, temporal_edge_w)))
		n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
		n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
		g_embed = global_max_pool(n_embed, batch_vec)

		#Process I3D feature
		img_feat = self.img_fc(img_feat)

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
    global_max_pool, 
    global_mean_pool,
    InstanceNorm
)
from torch.nn import Sequential, Linear, BatchNorm1d, LeakyReLU, Dropout, GRU, MultiheadAttention

class SpaceTempGoG_detr_dota(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=256, img_feat_dim=2048, num_classes=2):
        super(SpaceTempGoG_detr_dota, self).__init__()
        
        # Increased dimensions and added attention mechanisms
        self.num_heads = 4  # Increased from 1 to 4 for multi-head attention
        self.input_dim = input_dim
        
        # Enhanced object feature processing with residual connections
        self.x_fc = nn.Sequential(
            nn.Linear(self.input_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Improved label processing
        self.obj_l_fc = nn.Sequential(
            nn.Linear(300, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.2)
        )
        
        # Enhanced GNN components with edge feature processing
        self.edge_processor = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),
            nn.LeakyReLU(0.2)
        )
        
        # Spatial-temporal graph convolution with residual connections
        self.gc1_spatial = GATv2Conv(
            embedding_dim * 2 + embedding_dim, 
            embedding_dim // 2, 
            heads=self.num_heads,
            edge_dim=embedding_dim // 2
        )
        self.gc1_norm1 = InstanceNorm((embedding_dim // 2) * self.num_heads)
        
        self.gc1_temporal = GATv2Conv(
            embedding_dim * 2 + embedding_dim, 
            embedding_dim // 2, 
            heads=self.num_heads,
            edge_dim=embedding_dim // 2
        )
        self.gc1_norm2 = InstanceNorm((embedding_dim // 2) * self.num_heads)
        
        # Hierarchical pooling
        self.pool1 = TopKPooling(embedding_dim * self.num_heads, ratio=0.8)
        self.pool2 = TopKPooling(embedding_dim * self.num_heads // 2, ratio=0.7)
        
        # Enhanced I3D feature processing
        self.img_fc = nn.Sequential(
            nn.Linear(img_feat_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3)
        )
        
        # Multi-modal fusion with cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=self.num_heads,
            dropout=0.2
        )
        
        # Temporal modeling with GRU
        self.temporal_model = nn.GRU(
            input_size=embedding_dim * 2,
            hidden_size=embedding_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        # Enhanced classification head
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.BatchNorm1d(embedding_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim // 2, num_classes)
        )
        
        # Additional outputs for early warning
        self.early_warning = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(embedding_dim // 2, 1),
            nn.Sigmoid()
        )
        
        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, 
                edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):
        # Process object features with residual connection
        x_feat = self.x_fc(x[:, :self.input_dim])
        x_label = self.obj_l_fc(x[:, self.input_dim:])
        x = torch.cat((x_feat, x_label), 1)
        
        # Process edge features
        edge_emb = self.edge_processor(edge_embeddings[:, -1:])
        
        # Enhanced spatial-temporal graph processing
        n_embed_spatial = self.relu(self.gc1_norm1(
            self.gc1_spatial(x, edge_index, edge_attr=edge_emb)
        ))
        n_embed_temporal = self.relu(self.gc1_norm2(
            self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_emb)
        ))
        n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
        
        # Hierarchical graph pooling
        n_embed, edge_index, _, batch_vec, _, _ = self.pool1(n_embed, edge_index, None, batch_vec)
        n_embed, edge_index, _, batch_vec, _, _ = self.pool2(n_embed, edge_index, None, batch_vec)
        
        # Multi-level graph embedding
        g_embed_max = global_max_pool(n_embed, batch_vec)
        g_embed_mean = global_mean_pool(n_embed, batch_vec)
        g_embed = torch.cat((g_embed_max, g_embed_mean), 1)
        
        # Process I3D features
        img_feat = self.img_fc(img_feat)
        
        # Temporal modeling of I3D features
        img_feat, _ = self.temporal_model(img_feat.unsqueeze(0))
        img_feat = img_feat.squeeze(0)
        
        # Cross-attention between graph and visual features
        attn_out, _ = self.cross_attn(
            g_embed.unsqueeze(0),
            img_feat.unsqueeze(0),
            img_feat.unsqueeze(0)
        )
        fused_features = torch.cat((attn_out.squeeze(0), g_embed), 1)
        
        # Classification and early warning
        logits_mc = self.classifier(fused_features)
        probs_mc = self.softmax(logits_mc)
        warning_signal = self.early_warning(fused_features)
        
        return logits_mc, probs_mc, warning_signal
