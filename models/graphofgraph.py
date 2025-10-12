import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from torch.autograd import Variable
from torch_geometric.nn import GCNConv, global_max_pool, GATv2Conv, TopKPooling, SAGPooling
from torch_geometric.nn.norm import InstanceNorm
import copy
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_modules import EMSA, Memory_Attention_Aggregation, Auxiliary_Self_Attention_Aggregation

# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2, emsa_groups=4):
#         super(SpaceTempGoG_detr_dad, self).__init__()

#         # Make embedding divisible by EMSA groups
#         assert embedding_dim * 2 % emsa_groups == 0, f"concat_dim={embedding_dim*2} must be divisible by EMSA groups={emsa_groups}"
#         self.embedding_dim = embedding_dim
#         self.emsa_groups = emsa_groups

#         # Linear projections
#         self.obj_proj = nn.Linear(input_dim, embedding_dim)
#         self.global_proj = nn.Linear(img_feat_dim, embedding_dim)

#         concat_dim = embedding_dim * 2

#         # Parallel attention modules
#         self.memory_attention = Memory_Attention_Aggregation(agg_dim=concat_dim, d_model=concat_dim)
#         self.aux_attention = Auxiliary_Self_Attention_Aggregation(agg_dim=concat_dim)
#         self.temporal_emsa = EMSA(channels=concat_dim, factor=emsa_groups)

#         # Projection layers after attention outputs to unify shapes
#         self.mem_proj = nn.Linear(concat_dim, concat_dim)
#         self.aux_proj = nn.Linear(concat_dim, concat_dim)
#         self.emsa_proj = nn.Linear(concat_dim, concat_dim)

#         # Final classifier
#         concat_dim = 512
#         fused_dim = concat_dim * 3
#         self.classifier = nn.Sequential(
#             nn.Linear(fused_dim, fused_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(fused_dim // 2, num_classes)
#         )

#     def forward(self, obj_feats, global_feats):
#         # Ensure tensor dtype/device matches model
#         ref = next(self.parameters())
#         obj_feats = obj_feats.to(dtype=ref.dtype, device=ref.device)
#         global_feats = global_feats.to(dtype=ref.dtype, device=ref.device)

#         # Add batch dim if missing
#         if obj_feats.dim() == 2:
#             obj_feats = obj_feats.unsqueeze(0)
#         if global_feats.dim() == 2:
#             global_feats = global_feats.unsqueeze(0)

#         # Project features
#         obj_proj = self.obj_proj(obj_feats)        # [B, T_obj, embedding_dim]
#         global_proj = self.global_proj(global_feats)  # [B, T_global, embedding_dim]

#         # Align temporal dimension
#         T_max = max(obj_proj.size(1), global_proj.size(1))
#         if obj_proj.size(1) != T_max:
#             obj_proj = F.interpolate(obj_proj.transpose(1,2), size=T_max, mode='linear', align_corners=False).transpose(1,2)
#         if global_proj.size(1) != T_max:
#             global_proj = F.interpolate(global_proj.transpose(1,2), size=T_max, mode='linear', align_corners=False).transpose(1,2)

#         # Concatenate features
#         concat_feats = torch.cat([obj_proj, global_proj], dim=-1)  # [B, T_max, 2*embedding_dim]

#         # Apply attention modules
#         mem_out = self.mem_proj(self.memory_attention(concat_feats))  # [B, T_max, concat_dim]
#         aux_out_pre = self.aux_attention(concat_feats)  # [B, T_max, ?]

#         # Ensure aux_attention output has correct shape
#         concat_dim = self.embedding_dim * 2
#         if aux_out_pre.size(-1) != concat_dim:
#             # Project feature dimension to concat_dim, preserving temporal dimension
#             aux_out_pre = nn.Linear(aux_out_pre.size(-1), concat_dim).to(aux_out_pre.device)(aux_out_pre)
#         aux_out = self.aux_proj(aux_out_pre)  # [B, T_max, concat_dim]

#         # EMSA expects [B, C, H=1, W=T_max]
#         emsa_in = concat_feats.transpose(1,2).unsqueeze(2)  # [B, concat_dim, 1, T_max]
#         emsa_out = self.emsa_proj(self.temporal_emsa(emsa_in).squeeze(2).transpose(1,2))  # [B, T_max, concat_dim]

#         # ==== PRINT STATEMENTS ADDED ====
#         # print(f"obj_proj: {obj_proj.shape}, global_proj: {global_proj.shape}")
#         # print(f"concat_feats: {concat_feats.shape}")
#         # print(f"mem_out: {mem_out.shape}, aux_out: {aux_out.shape}, emsa_out: {emsa_out.shape}")
#         # ================================
		
#         # Concatenate all attention outputs
#         # fused = torch.cat([mem_out, aux_out, emsa_out], dim=-1)  # [B, T_max, 3*concat_dim]
#         emsa_out = emsa_out.squeeze(0)
#         # print(f"emsa_out after squeeze: {emsa_out.shape}")
#         # fused = torch.cat([mem_out, emsa_out], dim=-1)  # [B, T_max, 3*concat_dim]
#         # Expand aux_out to [1900, 512]
#         aux_out_expanded = aux_out.expand(mem_out.size(0), -1)

#         # print()
#         # print(f"mem_out: {mem_out.shape}, aux_out_expanded: {aux_out_expanded.shape}, emsa_out: {emsa_out.shape}")
		
#         # Concatenate along last dimension
#         fused = torch.cat([mem_out, aux_out_expanded, emsa_out], dim=-1)
#         # print(fused.shape)  # torch.Size([1900, 1536])

#         # Check if 1900 can be reshaped into 100 x 19
#         batch_size = 100
#         seq_len = 19
#         assert fused.size(0) == batch_size * seq_len, "1900 is not divisible by 100"

#         # Reshape
#         fused = fused.view(batch_size, seq_len, fused.size(1))  # [100, 19, 1536]

#         # # Add batch dimension
#         # fused = fused.unsqueeze(0)  # [1, 1900, 1536]
#         # print("fused shape after unsqueeze: ", fused.shape)
		
#         # Pool over temporal dimension
#         pooled = fused.mean(dim=1)  # [B, 3*concat_dim]
#         # print("pooled fused shape: ", fused.shape)  

#         # Classifier
#         logits_mc = self.classifier(pooled)
#         probs_mc = F.softmax(logits_mc, dim=-1)

#         return logits_mc, probs_mc



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

# # STAGNet
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
# 		# self.gc1_spatial = GCNConv(embedding_dim*2+embedding_dim//2, embedding_dim//2)  
		
# 	        # Improved GNN for encoding the object-level graph
# 		self.gc1_spatial = GATv2Conv(
# 		embedding_dim * 2 + embedding_dim // 2, 
# 		embedding_dim // 2, 
# 		heads=self.num_heads,
# 		edge_dim=1  # Using temporal_edge_w as edge features
# 		) 
# 		self.gc1_norm1 = InstanceNorm(embedding_dim//2)

# 		# self.gc1_temporal = GCNConv(embedding_dim*2+embedding_dim//2, embedding_dim//2)   
		
#         	# Improved temporal graph convolution
# 		self.gc1_temporal = GATv2Conv(
# 		embedding_dim * 2 + embedding_dim // 2, 
# 		embedding_dim // 2, 
# 		heads=self.num_heads,
# 		edge_dim=1  # Using temporal_edge_w as edge features
# 		)

# 		self.gc1_norm2 = InstanceNorm(embedding_dim//2)
# 		# self.pool = TopKPooling(embedding_dim, ratio=0.8)
# 		self.pool = SAGPooling(embedding_dim, ratio=0.8)

# 		#I3D features
# 		self.img_fc = nn.Linear(img_feat_dim, embedding_dim*2)

# 		# # Added LSTM for temporal sequence processing
# 		self.temporal_lstm = nn.LSTM(
# 		input_size=embedding_dim * 2,
# 		hidden_size=embedding_dim * 2,  # Changed to match input size
# 		num_layers=1,
# 		batch_first=True
# 		)

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
# 		# n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_weight=edge_embeddings[:, -1])))
	        
# 		# Improved Get graph embedding for object-level graph
# 		n_embed_spatial = self.relu(self.gc1_norm1(
# 		self.gc1_spatial(x, edge_index, edge_attr=edge_embeddings[:, -1].unsqueeze(1))
# 		))
		
# 		#old
# 		# n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, temporal_edge_w)))
		
# 		# Improved temporal processing
# 		temporal_edge_w = temporal_edge_w.to(x.dtype)
# 		n_embed_temporal = self.relu(self.gc1_norm2(
# 		self.gc1_temporal(x, temporal_adj_list, edge_attr=temporal_edge_w.unsqueeze(1))
# 		))
		
# 		n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
# 		n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
# 		g_embed = global_max_pool(n_embed, batch_vec)
		
# 		#Process I3D feature
# 		img_feat = self.img_fc(img_feat)
		
# 		# change - LSTM processing - reshape for temporal dimension
# 		img_feat = img_feat.unsqueeze(0)  # Add sequence dimension (1, num_nodes, features)
# 		img_feat, (_, _) = self.temporal_lstm(img_feat)  # Extract only output, discard hidden and cell state
# 		img_feat = img_feat.squeeze(0)  # Back to (num_nodes, features)	
		
# 		#Get frame embedding for all nodes in frame-level graph
# 		frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
# 		frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list)))
# 		frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img), 1)
# 		frame_embed_sg = self.relu(self.classify_fc1(frame_embed_))
# 		logits_mc = self.classify_fc2(frame_embed_sg)
# 		probs_mc = self.softmax(logits_mc)
		
# 		return logits_mc, probs_mc


import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention_modules import Memory_Attention_Aggregation, Auxiliary_Self_Attention_Aggregation, EMSA

# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dad, self).__init__()
		
#         # Linear projections for object and global features
#         self.obj_fc = nn.Linear(input_dim, embedding_dim)
#         self.global_fc = nn.Linear(img_feat_dim, embedding_dim)

#         concat_dim = embedding_dim * 2  # after concatenating obj + global

#         # Three parallel modules
#         self.memory_attention = Memory_Attention_Aggregation(agg_dim=concat_dim, d_model=concat_dim)
#         self.aux_attention = Auxiliary_Self_Attention_Aggregation(agg_dim=concat_dim)
#         self.temporal_emsa = EMSA(channels=concat_dim, factor=5)

#         # Final classifier after concatenating outputs of all three
#         fused_dim = concat_dim * 3
#         self.classifier = nn.Sequential(
#             nn.Linear(fused_dim, fused_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(fused_dim // 2, num_classes)
#         )

#     def forward(self, obj_feats, global_feats):
#         """
#         obj_feats: [B, T_obj, input_dim] or [T_obj, input_dim]
#         global_feats: [B, T_global, img_feat_dim] or [T_global, img_feat_dim]
#         """
#         # Add batch dimension if missing
#         if obj_feats.dim() == 2:
#             obj_feats = obj_feats.unsqueeze(0)  # [1, T_obj, input_dim]
#         if global_feats.dim() == 2:
#             global_feats = global_feats.unsqueeze(0)  # [1, T_global, img_feat_dim]

#         # Ensure same dtype
#         obj_feats = obj_feats.float()
#         global_feats = global_feats.float()
        
#         print(f"Input obj_feats: {obj_feats.shape}, global_feats: {global_feats.shape}")
    
#         # Step 1: project
#         obj_proj = self.obj_fc(obj_feats)           # [B, T_obj, embedding_dim]
#         global_proj = self.global_fc(global_feats)  # [B, T_global, embedding_dim]
#         print(f"After projection obj_proj: {obj_proj.shape}, global_proj: {global_proj.shape}")
    
#         # Step 2: align temporal dimension
#         T_obj = obj_proj.size(1)
#         T_global = global_proj.size(1)
#         T_max = max(T_obj, T_global)

#         if T_obj != T_max:
#             obj_proj = obj_proj.transpose(1, 2)  # [B, embedding_dim, T_obj]
#             obj_proj = F.interpolate(obj_proj, size=T_max, mode='linear', align_corners=False)
#             obj_proj = obj_proj.transpose(1, 2)
#             print(f"Interpolated obj_proj to: {obj_proj.shape}")

#         if T_global != T_max:
#             global_proj = global_proj.transpose(1, 2)  # [B, embedding_dim, T_global]
#             global_proj = F.interpolate(global_proj, size=T_max, mode='linear', align_corners=False)
#             global_proj = global_proj.transpose(1, 2)
#             print(f"Interpolated global_proj to: {global_proj.shape}")
    
#         # Step 3: concatenate along feature dimension
#         concat_feats = torch.cat([obj_proj, global_proj], dim=-1)  # [B, T_max, 2*embedding_dim]
#         print(f"Concatenated features shape: {concat_feats.shape}")
    
#         # Step 4: apply three attention modules in parallel
#         mem_out = self.memory_attention(concat_feats)
#         aux_out = self.aux_attention(concat_feats)

#         # For EMSA, reshape 3D [B, T, C] -> 4D [B, C, H=1, W=T]
#         emsa_in = concat_feats.transpose(1, 2).unsqueeze(2)  # [B, C, 1, T_max]
#         emsa_out = self.temporal_emsa(emsa_in)               # [B, C, 1, T_max]
#         emsa_out = emsa_out.squeeze(2).transpose(1, 2)      # back to [B, T_max, C]

#         print(f"mem_out: {mem_out.shape}, aux_out: {aux_out.shape}, emsa_out: {emsa_out.shape}")
    
#         # Step 5: concatenate outputs
#         fused = torch.cat([mem_out, aux_out, emsa_out], dim=-1)
#         print(f"Fused output shape: {fused.shape}")
    
#         # Step 6: pool over time
#         pooled = fused.mean(dim=1)
#         print(f"Pooled features shape: {pooled.shape}")
    
#         # Step 7: classifier
#         logits_mc = self.classifier(pooled)
#         probs_mc = F.softmax(logits_mc, dim=-1)
#         # print(f"Logits: {logits_mc.shape}, Probabilities: {probs_mc.shape}")
    
#         return logits_mc, probs_mc

# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2, emsa_factor=5):
#         super(SpaceTempGoG_detr_dad, self).__init__()
		
#         # Linear projections for object and global features
#         self.obj_fc = nn.Linear(input_dim, embedding_dim)
#         self.global_fc = nn.Linear(img_feat_dim, embedding_dim)

#         concat_dim = embedding_dim * 2  # after concatenating obj + global

#         # Three parallel modules
#         self.memory_attention = Memory_Attention_Aggregation(agg_dim=concat_dim, d_model=concat_dim)
#         self.aux_attention = Auxiliary_Self_Attention_Aggregation(agg_dim=concat_dim)

#         # --- Fix: adjust channels so they are divisible by emsa_factor ---
#         if concat_dim % emsa_factor != 0:
#             adjusted_dim = (concat_dim // emsa_factor) * emsa_factor
#             if adjusted_dim == 0:  # safeguard
#                 adjusted_dim = emsa_factor
#             self.proj_for_emsa = nn.Conv2d(concat_dim, adjusted_dim, kernel_size=1)
#             emsa_in_dim = adjusted_dim
#         else:
#             self.proj_for_emsa = None
#             emsa_in_dim = concat_dim

#         self.temporal_emsa = EMSA(channels=emsa_in_dim, factor=emsa_factor)

#         # Final classifier after concatenating outputs of all three
#         fused_dim = concat_dim * 3
#         self.classifier = nn.Sequential(
#             nn.Linear(fused_dim, fused_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(fused_dim // 2, num_classes)
#         )

#     def forward(self, obj_feats, global_feats):
#         """
#         obj_feats: [B, T_obj, input_dim] or [B, C, H, W]
#         global_feats: [B, T_global, img_feat_dim] or [B, C, H, W]
#         """

#         # ---- Case 1: If input is [B, C, H, W] (images), flatten spatial dims ----
#         if obj_feats.dim() == 4:  # [B, C, H, W]
#             B, C, H, W = obj_feats.shape
#             obj_feats = obj_feats.view(B, H * W, C)  # [B, T_obj, input_dim]
#             print(f"Reshaped obj_feats from [B,C,H,W] -> {obj_feats.shape}")

#         if global_feats.dim() == 4:  # [B, C, H, W]
#             B, C, H, W = global_feats.shape
#             global_feats = global_feats.view(B, H * W, C)  # [B, T_global, img_feat_dim]
#             print(f"Reshaped global_feats from [B,C,H,W] -> {global_feats.shape}")

#         # ---- Case 2: If input is [T, D], add batch dimension ----
#         if obj_feats.dim() == 2:
#             obj_feats = obj_feats.unsqueeze(0)
#         if global_feats.dim() == 2:
#             global_feats = global_feats.unsqueeze(0)

#         # Ensure float dtype
#         obj_feats = obj_feats.float()
#         global_feats = global_feats.float()
        
#         print(f"Input obj_feats: {obj_feats.shape}, global_feats: {global_feats.shape}")
    
#         # Step 1: project
#         obj_proj = self.obj_fc(obj_feats)           # [B, T_obj, embedding_dim]
#         global_proj = self.global_fc(global_feats)  # [B, T_global, embedding_dim]
#         print(f"After projection obj_proj: {obj_proj.shape}, global_proj: {global_proj.shape}")
    
#         # Step 2: align temporal dimension
#         T_obj = obj_proj.size(1)
#         T_global = global_proj.size(1)
#         T_max = max(T_obj, T_global)

#         if T_obj != T_max:
#             obj_proj = obj_proj.transpose(1, 2)  # [B, embedding_dim, T_obj]
#             obj_proj = F.interpolate(obj_proj, size=T_max, mode='linear', align_corners=False)
#             obj_proj = obj_proj.transpose(1, 2)
#             print(f"Interpolated obj_proj to: {obj_proj.shape}")

#         if T_global != T_max:
#             global_proj = global_proj.transpose(1, 2)  # [B, embedding_dim, T_global]
#             global_proj = F.interpolate(global_proj, size=T_max, mode='linear', align_corners=False)
#             global_proj = global_proj.transpose(1, 2)
#             print(f"Interpolated global_proj to: {global_proj.shape}")
    
#         # Step 3: concatenate along feature dimension
#         concat_feats = torch.cat([obj_proj, global_proj], dim=-1)  # [B, T_max, 2*embedding_dim]
#         print(f"Concatenated features shape: {concat_feats.shape}")
    
#         # Step 4: apply three attention modules in parallel
#         mem_out = self.memory_attention(concat_feats)
#         aux_out = self.aux_attention(concat_feats)

#         # For EMSA, reshape 3D [B, T, C] -> 4D [B, C, H=1, W=T]
#         emsa_in = concat_feats.transpose(1, 2).unsqueeze(2)  # [B, C, 1, T_max]

#         # --- Fix: project channels if needed ---
#         if self.proj_for_emsa is not None:
#             emsa_in = self.proj_for_emsa(emsa_in)

#         emsa_out = self.temporal_emsa(emsa_in)               # [B, C, 1, T_max]
#         emsa_out = emsa_out.squeeze(2).transpose(1, 2)      # back to [B, T_max, C]

#         print(f"mem_out: {mem_out.shape}, aux_out: {aux_out.shape}, emsa_out: {emsa_out.shape}")
    
#         # Step 5: concatenate outputs
#         fused = torch.cat([mem_out, aux_out, emsa_out], dim=-1)
#         print(f"Fused output shape: {fused.shape}")
    
#         # Step 6: pool over time
#         pooled = fused.mean(dim=1)
#         print(f"Pooled features shape: {pooled.shape}")
    
#         # Step 7: classifier
#         logits_mc = self.classifier(pooled)
#         probs_mc = F.softmax(logits_mc, dim=-1)
#         print(f"Logits: {logits_mc.shape}, Probabilities: {probs_mc.shape}")
    
#         return logits_mc, probs_mc


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .attention_modules import EMSA, Memory_Attention_Aggregation, Auxiliary_Self_Attention_Aggregation

# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2, emsa_groups=4):
#         super(SpaceTempGoG_detr_dad, self).__init__()

#         # Make embedding divisible by EMSA groups
#         assert embedding_dim * 2 % emsa_groups == 0, f"concat_dim={embedding_dim*2} must be divisible by EMSA groups={emsa_groups}"
#         self.embedding_dim = embedding_dim
#         self.emsa_groups = emsa_groups

#         # Linear projections
#         self.obj_proj = nn.Linear(input_dim, embedding_dim)
#         self.global_proj = nn.Linear(img_feat_dim, embedding_dim)

#         concat_dim = embedding_dim * 2

#         # Parallel attention modules
#         self.memory_attention = Memory_Attention_Aggregation(agg_dim=concat_dim, d_model=concat_dim)
#         self.aux_attention = Auxiliary_Self_Attention_Aggregation(agg_dim=concat_dim)
#         self.temporal_emsa = EMSA(channels=concat_dim, factor=emsa_groups)

#         # Projection layers after attention outputs to unify shapes
#         self.mem_proj = nn.Linear(concat_dim, concat_dim)
#         self.aux_proj = nn.Linear(concat_dim, concat_dim)
#         self.emsa_proj = nn.Linear(concat_dim, concat_dim)

#         # Final classifier
# 		concat_dim = 512
#         fused_dim = concat_dim * 3
#         self.classifier = nn.Sequential(
#             nn.Linear(fused_dim, fused_dim // 2),
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(fused_dim // 2, num_classes)
#         )

#     def forward(self, obj_feats, global_feats):
#         # Ensure tensor dtype/device matches model
#         ref = next(self.parameters())
#         obj_feats = obj_feats.to(dtype=ref.dtype, device=ref.device)
#         global_feats = global_feats.to(dtype=ref.dtype, device=ref.device)

#         # Add batch dim if missing
#         if obj_feats.dim() == 2:
#             obj_feats = obj_feats.unsqueeze(0)
#         if global_feats.dim() == 2:
#             global_feats = global_feats.unsqueeze(0)

#         # Project features
#         obj_proj = self.obj_proj(obj_feats)        # [B, T_obj, embedding_dim]
#         global_proj = self.global_proj(global_feats)  # [B, T_global, embedding_dim]

#         # Align temporal dimension
#         T_max = max(obj_proj.size(1), global_proj.size(1))
#         if obj_proj.size(1) != T_max:
#             obj_proj = F.interpolate(obj_proj.transpose(1,2), size=T_max, mode='linear', align_corners=False).transpose(1,2)
#         if global_proj.size(1) != T_max:
#             global_proj = F.interpolate(global_proj.transpose(1,2), size=T_max, mode='linear', align_corners=False).transpose(1,2)

#         # Concatenate features
#         concat_feats = torch.cat([obj_proj, global_proj], dim=-1)  # [B, T_max, 2*embedding_dim]

#         # Apply attention modules
#         mem_out = self.mem_proj(self.memory_attention(concat_feats))  # [B, T_max, concat_dim]
#         aux_out_pre = self.aux_attention(concat_feats)  # [B, T_max, ?]

#         # Ensure aux_attention output has correct shape
#         concat_dim = self.embedding_dim * 2
#         if aux_out_pre.size(-1) != concat_dim:
#             # Project feature dimension to concat_dim, preserving temporal dimension
#             aux_out_pre = nn.Linear(aux_out_pre.size(-1), concat_dim).to(aux_out_pre.device)(aux_out_pre)
#         aux_out = self.aux_proj(aux_out_pre)  # [B, T_max, concat_dim]

#         # EMSA expects [B, C, H=1, W=T_max]
#         emsa_in = concat_feats.transpose(1,2).unsqueeze(2)  # [B, concat_dim, 1, T_max]
#         emsa_out = self.emsa_proj(self.temporal_emsa(emsa_in).squeeze(2).transpose(1,2))  # [B, T_max, concat_dim]

#         # ==== PRINT STATEMENTS ADDED ====
#         print(f"obj_proj: {obj_proj.shape}, global_proj: {global_proj.shape}")
#         print(f"concat_feats: {concat_feats.shape}")
#         print(f"mem_out: {mem_out.shape}, aux_out: {aux_out.shape}, emsa_out: {emsa_out.shape}")
#         # ================================
		
#         # Concatenate all attention outputs
#         # fused = torch.cat([mem_out, aux_out, emsa_out], dim=-1)  # [B, T_max, 3*concat_dim]
#         emsa_out = emsa_out.squeeze(0)
#         print(f"emsa_out after squeeze: {emsa_out.shape}")
#         # fused = torch.cat([mem_out, emsa_out], dim=-1)  # [B, T_max, 3*concat_dim]
#         # Expand aux_out to [1900, 512]
#         aux_out_expanded = aux_out.expand(mem_out.size(0), -1)
		
#         # Concatenate along last dimension
#         fused = torch.cat([mem_out, aux_out_expanded, emsa_out], dim=-1)
#         print(fused.shape)  # torch.Size([1900, 1536])

#         # Pool over temporal dimension
#         pooled = fused.mean(dim=1)  # [B, 3*concat_dim]

#         # Classifier
#         logits_mc = self.classifier(pooled)
#         probs_mc = F.softmax(logits_mc, dim=-1)

#         return logits_mc, probs_mc


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .attention_modules import EMSA, Memory_Attention_Aggregation, Auxiliary_Self_Attention_Aggregation

# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2, emsa_groups=4):
#         super(SpaceTempGoG_detr_dad, self).__init__()

#         # Make embedding divisible by EMSA groups
#         assert embedding_dim * 2 % emsa_groups == 0, f"concat_dim={embedding_dim*2} must be divisible by EMSA groups={emsa_groups}"
#         self.embedding_dim = embedding_dim
#         self.emsa_groups = emsa_groups

#         # Linear projections
#         self.obj_proj = nn.Linear(input_dim, embedding_dim)
#         self.global_proj = nn.Linear(img_feat_dim, embedding_dim)

#         concat_dim = embedding_dim * 2

#         # Parallel attention modules
#         self.memory_attention = Memory_Attention_Aggregation(agg_dim=concat_dim, d_model=concat_dim)
#         self.aux_attention = Auxiliary_Self_Attention_Aggregation(agg_dim=concat_dim)
#         self.temporal_emsa = EMSA(channels=concat_dim, factor=emsa_groups)

#         # Projection layers after attention outputs to unify shapes
#         self.mem_proj = nn.Linear(concat_dim, concat_dim)
#         self.aux_proj = nn.Linear(concat_dim, concat_dim)
#         self.emsa_proj = nn.Linear(concat_dim, concat_dim)

#         # Final classifier
#         fused_dim = concat_dim * 3  # Not used directly anymore, replaced with flattened fused size
#         self.classifier = nn.Sequential(
#             nn.Linear(3 * concat_dim * 1000, fused_dim // 2),  # Use a placeholder for now; adjust after knowing T_max
#             nn.ReLU(inplace=True),
#             nn.Dropout(0.5),
#             nn.Linear(fused_dim // 2, num_classes)
#         )

#     def forward(self, obj_feats, global_feats):
#         # Ensure tensor dtype/device matches model
#         ref = next(self.parameters())
#         obj_feats = obj_feats.to(dtype=ref.dtype, device=ref.device)
#         global_feats = global_feats.to(dtype=ref.dtype, device=ref.device)

#         # Add batch dim if missing
#         if obj_feats.dim() == 2:
#             obj_feats = obj_feats.unsqueeze(0)
#         if global_feats.dim() == 2:
#             global_feats = global_feats.unsqueeze(0)

#         # Project features
#         obj_proj = self.obj_proj(obj_feats)        # [B, T_obj, embedding_dim]
#         global_proj = self.global_proj(global_feats)  # [B, T_global, embedding_dim]

#         # Align temporal dimension
#         T_max = max(obj_proj.size(1), global_proj.size(1))
#         if obj_proj.size(1) != T_max:
#             obj_proj = F.interpolate(obj_proj.transpose(1,2), size=T_max, mode='linear', align_corners=False).transpose(1,2)
#         if global_proj.size(1) != T_max:
#             global_proj = F.interpolate(global_proj.transpose(1,2), size=T_max, mode='linear', align_corners=False).transpose(1,2)

#         # Concatenate features
#         concat_feats = torch.cat([obj_proj, global_proj], dim=-1)  # [B, T_max, 2*embedding_dim]

#         # Apply attention modules
#         mem_out = self.mem_proj(self.memory_attention(concat_feats))  # [B, T_max, concat_dim]
#         aux_out_pre = self.aux_attention(concat_feats)  # [B, T_max, ?]

#         # Ensure aux_attention output has correct shape
#         concat_dim = self.embedding_dim * 2
#         if aux_out_pre.size(-1) != concat_dim:
#             aux_out_pre = nn.Linear(aux_out_pre.size(-1), concat_dim).to(aux_out_pre.device)(aux_out_pre)
#         aux_out = self.aux_proj(aux_out_pre)  # [B, T_max, concat_dim]

#         # EMSA expects [B, C, H=1, W=T_max]
#         emsa_in = concat_feats.transpose(1,2).unsqueeze(2)  # [B, concat_dim, 1, T_max]
#         emsa_out = self.emsa_proj(self.temporal_emsa(emsa_in).squeeze(2).transpose(1,2))  # [B, T_max, concat_dim]

#         # ==== PRINT STATEMENTS ====
#         print(f"obj_proj: {obj_proj.shape}, global_proj: {global_proj.shape}")
#         print(f"concat_feats: {concat_feats.shape}")
#         print(f"mem_out: {mem_out.shape}, aux_out: {aux_out.shape}, emsa_out: {emsa_out.shape}")
#         # ==========================

#         # Flatten all attention outputs
#         mem_flat = mem_out.flatten(start_dim=1)
#         aux_flat = aux_out.flatten(start_dim=1)
#         emsa_flat = emsa_out.flatten(start_dim=1)

#         # fused = torch.cat([mem_flat, aux_flat, emsa_flat], dim=-1)
#         fused = torch.cat([mem_flat, emsa_flat], dim=-1)
#         print(f"fused (1D per sample) shape: {fused.shape}")

#         # Classifier
#         logits_mc = self.classifier(fused)
#         probs_mc = F.softmax(logits_mc, dim=-1)

#         return logits_mc, probs_mc


# # Original Graph
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


# STAGNet - DoTA DADA
# from torch_geometric.nn import (
#     GATv2Conv, 
#     TopKPooling,
#     SAGPooling,
#     global_max_pool, 
#     global_mean_pool,
#     InstanceNorm
# )
# from torch.nn import Sequential, Linear, BatchNorm1d, LeakyReLU, Dropout, GRU, MultiheadAttention

# class SpaceTempGoG_detr_dota(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dota, self).__init__()

#         self.num_heads = 1
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # process the object graph features
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)
        
#         # Improved GNN for encoding the object-level graph
#         # self.gc1_spatial = GATv2Conv(
#         #     embedding_dim * 2 + embedding_dim // 2, 
#         #     embedding_dim // 2, 
#         #     heads=self.num_heads,
#         #     edge_dim=1  # Using temporal_edge_w as edge features
#         # )
#         # GNN for encoding the object-level graph
#         self.gc1_spatial = GCNConv(embedding_dim * 2 + embedding_dim // 2, embedding_dim // 2)
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2)
        
#         # Improved temporal graph convolution
#         self.gc1_temporal = GATv2Conv(
#             embedding_dim * 2 + embedding_dim // 2, 
#             embedding_dim // 2, 
#             heads=self.num_heads,
#             edge_dim=1  # Using temporal_edge_w as edge features
#         )
#         # self.gc1_temporal = GCNConv(embedding_dim * 2 + embedding_dim // 2, embedding_dim // 2)
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2)  # Removed *num_heads since we're using 1 head
        
#         # self.pool = TopKPooling(embedding_dim, ratio=0.8)
#         self.pool = SAGPooling(embedding_dim, ratio=0.8)

#         # I3D features with temporal processing
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
        
#         # # Added GRU for temporal sequence processing
#         # self.temporal_gru = nn.GRU(
#         #     input_size=embedding_dim * 2,
#         #     hidden_size=embedding_dim * 2,  # Changed to match input size
#         #     num_layers=1,
#         #     batch_first=True
#         # )

#         # Added LSTM for temporal sequence processing
#         self.temporal_lstm = nn.LSTM(
#             input_size=embedding_dim * 2,
#             hidden_size=embedding_dim * 2,  # Changed to match input size
#             num_layers=1,
#             batch_first=True
#         )

#         # Fixed dimension mismatches in these layers
#         self.gc2_sg = GATv2Conv(
#             embedding_dim,  # Input from g_embed
#             embedding_dim // 2, 
#             heads=self.num_heads
#         )
#         self.gc2_norm1 = InstanceNorm(embedding_dim // 2)
        
#         self.gc2_i3d = GATv2Conv(
#             embedding_dim * 2,  # Input from GRU output
#             embedding_dim // 2, 
#             heads=self.num_heads
#         )
#         self.gc2_norm2 = InstanceNorm(embedding_dim // 2)

#         self.classify_fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
#         self.classify_fc2 = nn.Linear(embedding_dim // 2, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):
#         # process object graph features
#         x_feat = self.x_fc(x[:, :self.input_dim])
#         x_feat = self.relu(self.x_bn1(x_feat))
#         x_label = self.obj_l_fc(x[:, self.input_dim:])
#         x_label = self.relu(self.obj_l_bn1(x_label))
#         x = torch.cat((x_feat, x_label), 1)

#         # Old Get graph embedding for object-level graph
#         n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_weight=edge_embeddings[:, -1])))
        
#         # Improved Get graph embedding for object-level graph
#         # n_embed_spatial = self.relu(self.gc1_norm1(
#         #     self.gc1_spatial(x, edge_index, edge_attr=edge_embeddings[:, -1].unsqueeze(1))
#         # ))
        
#         # Old temporal processing
#         # n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, temporal_edge_w)))
        
#         # Improved temporal processing
#         n_embed_temporal = self.relu(self.gc1_norm2(
#             self.gc1_temporal(x, temporal_adj_list, edge_attr=temporal_edge_w.unsqueeze(1))
#         ))
        
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)

#         # Process I3D feature with temporal modeling
#         img_feat = self.img_fc(img_feat)
#         # print("After img_fc:", img_feat.shape)
        
#         # GRU processing - reshape for temporal dimension
#         # img_feat = img_feat.unsqueeze(0)  # Add sequence dimension (1, num_nodes, features)
#         # img_feat, _ = self.temporal_gru(img_feat)
#         # img_feat = img_feat.squeeze(0)  # Back to (num_nodes, features)

# 		# LSTM processing - reshape for temporal dimension
#         img_feat = img_feat.unsqueeze(0)  # Add sequence dimension (1, num_nodes, features)
#         img_feat, (_, _) = self.temporal_lstm(img_feat)  # Extract only output, discard hidden and cell state
#         img_feat = img_feat.squeeze(0)  # Back to (num_nodes, features)

#         # Get frame embedding for all nodes in frame-level graph
#         frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
#         frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list)))
#         frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img), 1)
#         frame_embed_sg = self.relu(self.classify_fc1(frame_embed_))
#         logits_mc = self.classify_fc2(frame_embed_sg)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc


# # for transformer

# from torch_geometric.nn import (
#     GATv2Conv, 
#     TopKPooling,
#     SAGPooling,
#     global_max_pool, 
#     global_mean_pool,
#     InstanceNorm
# )
# from torch.nn import Sequential, Linear, BatchNorm1d, LeakyReLU, Dropout, GRU, MultiheadAttention
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class SpaceTempGoG_detr_dota(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dota, self).__init__()

#         self.num_heads = 1
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # process the object graph features
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)
        
#         # Improved GNN for encoding the object-level graph
#         self.gc1_spatial = GATv2Conv(
#             embedding_dim * 2 + embedding_dim // 2, 
#             embedding_dim // 2, 
#             heads=self.num_heads,
#             edge_dim=1  # Using temporal_edge_w as edge features
#         )
#         # GNN for encoding the object-level graph
#         # self.gc1_spatial = GCNConv(embedding_dim * 2 + embedding_dim // 2, embedding_dim // 2)
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2)
        
#         # Improved temporal graph convolution
#         self.gc1_temporal = GATv2Conv(
#             embedding_dim * 2 + embedding_dim // 2, 
#             embedding_dim // 2, 
#             heads=self.num_heads,
#             edge_dim=1  # Using temporal_edge_w as edge features
#         )
#         # self.gc1_temporal = GCNConv(embedding_dim * 2 + embedding_dim // 2, embedding_dim // 2)
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2)  # Removed *num_heads since we're using 1 head
        
#         # self.pool = TopKPooling(embedding_dim, ratio=0.8)
#         self.pool = SAGPooling(embedding_dim, ratio=0.8)

#         # I3D features with temporal processing
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
        
#         # # Added GRU for temporal sequence processing
#         # self.temporal_gru = nn.GRU(
#         #     input_size=embedding_dim * 2,
#         #     hidden_size=embedding_dim * 2,  # Changed to match input size
#         #     num_layers=1,
#         #     batch_first=True
#         # )

#         # Added LSTM for temporal sequence processing
#         # self.temporal_lstm = nn.LSTM(
#         #     input_size=embedding_dim * 2,
#         #     hidden_size=embedding_dim * 2,  # Changed to match input size
#         #     num_layers=1,
#         #     batch_first=True
#         # )

#         encoder_layer = TransformerEncoderLayer(d_model=embedding_dim*2, nhead=4, batch_first=True)
#         self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=2)

#         # Fixed dimension mismatches in these layers
#         self.gc2_sg = GATv2Conv(
#             embedding_dim,  # Input from g_embed
#             embedding_dim // 2, 
#             heads=self.num_heads
#         )
#         self.gc2_norm1 = InstanceNorm(embedding_dim // 2)
        
#         self.gc2_i3d = GATv2Conv(
#             embedding_dim * 2,  # Input from GRU output
#             embedding_dim // 2, 
#             heads=self.num_heads
#         )
#         self.gc2_norm2 = InstanceNorm(embedding_dim // 2)

#         self.classify_fc1 = nn.Linear(embedding_dim, embedding_dim // 2)
#         self.classify_fc2 = nn.Linear(embedding_dim // 2, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings, temporal_adj_list, temporal_edge_w, batch_vec):
#         # process object graph features
#         x_feat = self.x_fc(x[:, :self.input_dim])
#         x_feat = self.relu(self.x_bn1(x_feat))
#         x_label = self.obj_l_fc(x[:, self.input_dim:])
#         x_label = self.relu(self.obj_l_bn1(x_label))
#         x = torch.cat((x_feat, x_label), 1)

#         # Old Get graph embedding for object-level graph
#         # n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_weight=edge_embeddings[:, -1])))
        
#         # Improved Get graph embedding for object-level graph
#         n_embed_spatial = self.relu(self.gc1_norm1(
#             self.gc1_spatial(x, edge_index, edge_attr=edge_embeddings[:, -1].unsqueeze(1))
#         ))
        
#         # Old temporal processing
#         # n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, temporal_edge_w)))
        
#         # Improved temporal processing
#         n_embed_temporal = self.relu(self.gc1_norm2(
#             self.gc1_temporal(x, temporal_adj_list, edge_attr=temporal_edge_w.unsqueeze(1))
#         ))
        
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)

#         # Process I3D feature with temporal modeling
#         img_feat = self.img_fc(img_feat)
#         # print("After img_fc:", img_feat.shape)
        
#         # GRU processing - reshape for temporal dimension
#         # img_feat = img_feat.unsqueeze(0)  # Add sequence dimension (1, num_nodes, features)
#         # img_feat, _ = self.temporal_gru(img_feat)
#         # img_feat = img_feat.squeeze(0)  # Back to (num_nodes, features)

# 		# LSTM processing - reshape for temporal dimension
#         # img_feat = img_feat.unsqueeze(0)  # Add sequence dimension (1, num_nodes, features)
#         # img_feat, (_, _) = self.temporal_lstm(img_feat)  # Extract only output, discard hidden and cell state
#         # img_feat = img_feat.squeeze(0)  # Back to (num_nodes, features)

#         # Transformer
#         img_feat = img_feat.unsqueeze(0)  
#         img_feat = self.temporal_transformer(img_feat)  
#         img_feat = img_feat.squeeze(0)

#         # Get frame embedding for all nodes in frame-level graph
#         frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
#         frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list)))
#         frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img), 1)
#         frame_embed_sg = self.relu(self.classify_fc1(frame_embed_))
#         logits_mc = self.classify_fc2(frame_embed_sg)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc


# This gave DoTa 95
# # filename: space_temp_gog_detr_dota_transformer.py
# import torch
# import torch.nn as nn
# from torch_geometric.nn import (
#     TransformerConv,
#     SAGPooling,
#     global_max_pool,
#     InstanceNorm
# )
# from torch.nn import TransformerEncoder, TransformerEncoderLayer


# class SpaceTempGoG_detr_dota(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dota, self).__init__()

#         self.num_heads = 4
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # process the object graph features
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)   # 2048 -> 256
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)         # 300 -> 64
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

#         # Graph Transformer for spatial graph
#         self.gc1_spatial = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,   # 256 + 64 = 320
#             out_channels=embedding_dim // 2,                      # 64
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph Transformer for temporal graph
#         self.gc1_temporal = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph pooling
#         self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

#         # I3D features -> Transformer
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)   # 2048 -> 256
#         encoder_layer = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True
#         )
#         self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=2)

#         # Frame-level graph encoding
#         self.gc2_sg = TransformerConv(
#             in_channels=embedding_dim * self.num_heads,  # from g_embed
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc2_i3d = TransformerConv(
#             in_channels=embedding_dim * 2,  # from Transformer
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # ---- FIX: determine concat dimension correctly ----
#         concat_dim = (embedding_dim // 2 * self.num_heads) + (embedding_dim // 2 * self.num_heads)
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings,
#                 temporal_adj_list, temporal_edge_w, batch_vec):

#         # process object graph features
#         x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
#         x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
#         x = torch.cat((x_feat, x_label), 1)  # (N, 320)

#         # spatial graph
#         n_embed_spatial = self.relu(self.gc1_norm1(
#             self.gc1_spatial(x, edge_index, edge_attr=edge_embeddings[:, -1].unsqueeze(1))
#         ))

#         # temporal graph
#         n_embed_temporal = self.relu(self.gc1_norm2(
#             self.gc1_temporal(x, temporal_adj_list, edge_attr=temporal_edge_w.unsqueeze(1))
#         ))

#         # concat + pooling
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)

#         # process I3D features with Transformer
#         img_feat = self.img_fc(img_feat)              # (B, 256)
#         img_feat = img_feat.unsqueeze(0)              # (1, B, 256)
#         img_feat = self.temporal_transformer(img_feat)  # (1, B, 256)
#         img_feat = img_feat.squeeze(0)                # (B, 256)

#         # frame-level embeddings
#         frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
#         frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list)))

#         # concat
#         frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img), 1)

#         # classification
#         frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
#         logits_mc = self.classify_fc2(frame_embed_)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc

# This Gave 67 and 68 for DAD dataset for orig and slowfast respectively
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import (
#     TransformerConv,
#     SAGPooling,
#     global_max_pool,
#     InstanceNorm
# )
# from torch.nn import TransformerEncoder, TransformerEncoderLayer


# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dad, self).__init__()

#         self.num_heads = 4
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # process the object graph features
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)   # 2048 -> 256
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)         # 300 -> 64
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

#         # Graph Transformer for spatial graph
#         self.gc1_spatial = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,   # 256 + 64 = 320
#             out_channels=embedding_dim // 2,                      # 64
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph Transformer for temporal graph
#         self.gc1_temporal = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph pooling
#         self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

#         # I3D features -> Transformer
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)   # 2048 -> 256
#         encoder_layer = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True
#         )
#         self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=2)

#         # Frame-level graph encoding
#         self.gc2_sg = TransformerConv(
#             in_channels=embedding_dim * self.num_heads,  # from g_embed
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc2_i3d = TransformerConv(
#             in_channels=embedding_dim * 2,  # from Transformer
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # ---- FIX: determine concat dimension correctly ----
#         concat_dim = (embedding_dim // 2 * self.num_heads) + (embedding_dim // 2 * self.num_heads)
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings,
#                 temporal_adj_list, temporal_edge_w, batch_vec):

#         # process object graph features
#         x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
#         x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
#         x = torch.cat((x_feat, x_label), 1)  # (N, 320)

#         # spatial graph
#         edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_spatial = self.relu(self.gc1_norm1(
#             self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
#         ))

#         # temporal graph
#         edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_temporal = self.relu(self.gc1_norm2(
#             self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
#         ))

#         # concat + pooling
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)

#         # process I3D features with Transformer
#         img_feat = self.img_fc(img_feat)              # (B, 256)
#         img_feat = img_feat.unsqueeze(0)              # (1, B, 256)
#         img_feat = self.temporal_transformer(img_feat)  # (1, B, 256)
#         img_feat = img_feat.squeeze(0)                # (B, 256)

#         # frame-level embeddings
#         frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
#         frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list)))

#         # concat
#         frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img), 1)

#         # classification
#         frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
#         logits_mc = self.classify_fc2(frame_embed_)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc


# This gave 75 on orig dataset, parallel two transformer
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import (
#     TransformerConv,
#     SAGPooling,
#     global_max_pool,
#     InstanceNorm
# )
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dad, self).__init__()

#         self.num_heads = 4
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # -----------------------
#         # Object graph features
#         # -----------------------
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)   # 2048 -> 256
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)         # 300 -> 64
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

#         # -----------------------
#         # Spatial and temporal graph transformers
#         # -----------------------
#         self.gc1_spatial = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,  # 256+64=320
#             out_channels=embedding_dim // 2,                      # 64
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc1_temporal = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph pooling
#         self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

#         # -----------------------
#         # I3D features -> Transformer
#         # -----------------------
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)  # 2048 -> 256
#         encoder_layer = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True
#         )
#         self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=2)

#         # Parallel TemporalFusionTransformer branch
#         encoder_layer_fusion = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True,
#             dropout=0.1
#         )
#         self.temporal_fusion_transformer = TransformerEncoder(encoder_layer_fusion, num_layers=2)

#         # -----------------------
#         # Frame-level graph encoding
#         # -----------------------
#         self.gc2_sg = TransformerConv(
#             in_channels=embedding_dim * self.num_heads,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc2_i3d = TransformerConv(
#             in_channels=embedding_dim * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # -----------------------
#         # Classification
#         # -----------------------
#         concat_dim = (embedding_dim // 2 * self.num_heads) + \
#                      (embedding_dim // 2 * self.num_heads) + \
#                      (embedding_dim * 2)  # adding temporal fusion branch
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings,
#                 temporal_adj_list, temporal_edge_w, batch_vec):

#         # -----------------------
#         # Object graph processing
#         # -----------------------
#         x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
#         x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
#         x = torch.cat((x_feat, x_label), 1)  # (N, 320)

#         # Spatial graph
#         edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_spatial = self.relu(self.gc1_norm1(
#             self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
#         ))

#         # Temporal graph
#         edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_temporal = self.relu(self.gc1_norm2(
#             self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
#         ))

#         # Concat + pooling
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)

#         # -----------------------
#         # I3D feature processing
#         # -----------------------
#         # Original Transformer
#         img_feat_orig = self.img_fc(img_feat).unsqueeze(0)
#         img_feat_orig = self.temporal_transformer(img_feat_orig)
#         img_feat_orig = img_feat_orig.squeeze(0)

#         # Parallel TemporalFusionTransformer
#         img_feat_fusion = self.img_fc(img_feat).unsqueeze(0)
#         img_feat_fusion = self.temporal_fusion_transformer(img_feat_fusion)
#         img_feat_fusion = img_feat_fusion.squeeze(0)

#         # -----------------------
#         # Frame-level embeddings
#         # -----------------------
#         frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
#         frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat_orig, video_adj_list)))

#         # Concatenate all features
#         frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img, img_feat_fusion), 1)

#         # -----------------------
#         # Classification
#         # -----------------------
#         frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
#         logits_mc = self.classify_fc2(frame_embed_)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc

# same as above but is_caussal added
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import (
#     TransformerConv,
#     SAGPooling,
#     global_max_pool,
#     InstanceNorm
# )
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dad, self).__init__()

#         self.num_heads = 4
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # -----------------------
#         # Object graph features
#         # -----------------------
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

#         # -----------------------
#         # Spatial and temporal graph transformers
#         # -----------------------
#         self.gc1_spatial = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc1_temporal = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph pooling
#         self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

#         # -----------------------
#         # I3D features -> Transformer (causal)
#         # -----------------------
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
#         encoder_layer = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True
#         )
#         self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=2)

#         encoder_layer_fusion = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True,
#             dropout=0.1
#         )
#         self.temporal_fusion_transformer = TransformerEncoder(encoder_layer_fusion, num_layers=2)

#         # -----------------------
#         # Frame-level graph encoding
#         # -----------------------
#         self.gc2_sg = TransformerConv(
#             in_channels=embedding_dim * self.num_heads,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc2_i3d = TransformerConv(
#             in_channels=embedding_dim * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # -----------------------
#         # Classification
#         # -----------------------
#         concat_dim = (embedding_dim // 2 * self.num_heads) + \
#                      (embedding_dim // 2 * self.num_heads) + \
#                      (embedding_dim * 2)
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings,
#                 temporal_adj_list, temporal_edge_w, batch_vec):

#         # -----------------------
#         # Object graph processing
#         # -----------------------
#         x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
#         x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
#         x = torch.cat((x_feat, x_label), 1)

#         # Spatial graph
#         edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_spatial = self.relu(self.gc1_norm1(
#             self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
#         ))

#         # Temporal graph (causal)
#         edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_temporal = self.relu(self.gc1_norm2(
#             self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
#         ))

#         # Concat + pooling
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)

#         # -----------------------
#         # I3D feature processing (causal)
#         # -----------------------
#         img_feat_proj = self.img_fc(img_feat).unsqueeze(0)  # (1, seq_len, d_model)
#         img_feat_orig = self.temporal_transformer(img_feat_proj, is_causal=True).squeeze(0)
#         img_feat_fusion = self.temporal_fusion_transformer(img_feat_proj, is_causal=True).squeeze(0)

#         # -----------------------
#         # Frame-level embeddings
#         # -----------------------
#         frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
#         frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat_orig, video_adj_list)))

#         # Concatenate all features
#         frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img, img_feat_fusion), 1)

#         # -----------------------
#         # Classification
#         # -----------------------
#         frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
#         logits_mc = self.classify_fc2(frame_embed_)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc


# import torch
# import torch.nn as nn
# from torch_geometric.nn import (
#     TransformerConv,
#     SAGPooling,
#     global_max_pool,
#     InstanceNorm
# )
# from torch.nn import TransformerEncoder, TransformerEncoderLayer


# class SpaceTempGoG_detr_dota(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dota, self).__init__()

#         self.num_heads = 4
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # -----------------------
#         # Object graph features
#         # -----------------------
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

#         # -----------------------
#         # Spatial and temporal graph transformers
#         # -----------------------
#         self.gc1_spatial = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc1_temporal = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph pooling
#         self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

#         # -----------------------
#         # I3D features -> Transformer
#         # -----------------------
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
#         encoder_layer = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True
#         )
#         self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=2)

#         # Parallel TemporalFusionTransformer
#         encoder_layer_fusion = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True,
#             dropout=0.1
#         )
#         self.temporal_fusion_transformer = TransformerEncoder(encoder_layer_fusion, num_layers=2)

#         # -----------------------
#         # Frame-level graph encoding
#         # -----------------------
#         self.gc2_sg = TransformerConv(
#             in_channels=embedding_dim * self.num_heads,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc2_i3d = TransformerConv(
#             in_channels=embedding_dim * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # -----------------------
#         # Classification
#         # -----------------------
#         concat_dim = (embedding_dim // 2 * self.num_heads) + \
#                      (embedding_dim // 2 * self.num_heads) + \
#                      (embedding_dim * 2)  # adding TemporalFusionTransformer branch
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings,
#                 temporal_adj_list, temporal_edge_w, batch_vec):

#         # -----------------------
#         # Object graph processing
#         # -----------------------
#         x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
#         x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
#         x = torch.cat((x_feat, x_label), 1)

#         # Spatial graph
#         edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_spatial = self.relu(self.gc1_norm1(
#             self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
#         ))

#         # Temporal graph
#         edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_temporal = self.relu(self.gc1_norm2(
#             self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
#         ))

#         # Concat + pooling
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)

#         # -----------------------
#         # I3D feature processing
#         # -----------------------
#         # Original Transformer
#         img_feat_orig = self.img_fc(img_feat).unsqueeze(0)
#         img_feat_orig = self.temporal_transformer(img_feat_orig)
#         img_feat_orig = img_feat_orig.squeeze(0)

#         # Parallel TemporalFusionTransformer
#         img_feat_fusion = self.img_fc(img_feat).unsqueeze(0)
#         img_feat_fusion = self.temporal_fusion_transformer(img_feat_fusion)
#         img_feat_fusion = img_feat_fusion.squeeze(0)

#         # -----------------------
#         # Frame-level embeddings
#         # -----------------------
#         frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
#         frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat_orig, video_adj_list)))

#         # Concatenate all features
#         frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img, img_feat_fusion), 1)

#         # -----------------------
#         # Classification
#         # -----------------------
#         frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
#         logits_mc = self.classify_fc2(frame_embed_)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc

# 3 Transformers are used
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import (
#     TransformerConv,
#     SAGPooling,
#     global_max_pool,
#     InstanceNorm
# )
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dad, self).__init__()

#         self.num_heads = 4
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # -----------------------
#         # Object graph features
#         # -----------------------
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)   # 2048 -> 256
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)         # 300 -> 64
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

#         # -----------------------
#         # Spatial and temporal graph transformers
#         # -----------------------
#         self.gc1_spatial = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,  # 256+64=320
#             out_channels=embedding_dim // 2,                      # 64
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc1_temporal = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph pooling
#         self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

#         # -----------------------
#         # I3D features -> Transformers
#         # -----------------------
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)  # 2048 -> 256

#         # Original Transformer branch
#         encoder_layer = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True
#         )
#         self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=2)

#         # Parallel Fusion Transformer branch
#         encoder_layer_fusion = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True,
#             dropout=0.1
#         )
#         self.temporal_fusion_transformer = TransformerEncoder(encoder_layer_fusion, num_layers=2)

#         # NEW: Perceiver-style transformer branch
#         encoder_layer_perceiver = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=8,                # deeper attention
#             batch_first=True,
#             dropout=0.2
#         )
#         self.temporal_perceiver = TransformerEncoder(encoder_layer_perceiver, num_layers=3)

#         # -----------------------
#         # Frame-level graph encoding
#         # -----------------------
#         self.gc2_sg = TransformerConv(
#             in_channels=embedding_dim * self.num_heads,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc2_i3d = TransformerConv(
#             in_channels=embedding_dim * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # -----------------------
#         # Classification
#         # -----------------------
#         concat_dim = (embedding_dim // 2 * self.num_heads) + \
#                      (embedding_dim // 2 * self.num_heads) + \
#                      (embedding_dim * 2) + \
#                      (embedding_dim * 2)   # <-- added perceiver branch
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings,
#                 temporal_adj_list, temporal_edge_w, batch_vec):

#         # -----------------------
#         # Object graph processing
#         # -----------------------
#         x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
#         x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
#         x = torch.cat((x_feat, x_label), 1)  # (N, 320)

#         # Spatial graph
#         edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_spatial = self.relu(self.gc1_norm1(
#             self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
#         ))

#         # Temporal graph
#         edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_temporal = self.relu(self.gc1_norm2(
#             self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
#         ))

#         # Concat + pooling
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)

#         # -----------------------
#         # I3D feature processing (3 parallel branches)
#         # -----------------------
#         img_feat_proj = self.img_fc(img_feat).unsqueeze(0)

#         # Branch 1: Original Transformer
#         img_feat_orig = self.temporal_transformer(img_feat_proj).squeeze(0)

#         # Branch 2: Fusion Transformer
#         img_feat_fusion = self.temporal_fusion_transformer(img_feat_proj).squeeze(0)

#         # Branch 3: Perceiver Transformer
#         img_feat_perceiver = self.temporal_perceiver(img_feat_proj).squeeze(0)

#         # -----------------------
#         # Frame-level embeddings
#         # -----------------------
#         frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
#         frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat_orig, video_adj_list)))

#         # Concatenate all features
#         frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img,
#                                   img_feat_fusion, img_feat_perceiver), 1)

#         # -----------------------
#         # Classification
#         # -----------------------
#         frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
#         logits_mc = self.classify_fc2(frame_embed_)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import (
#     TransformerConv,
#     SAGPooling,
#     global_max_pool,
#     InstanceNorm
# )
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# # -----------------------
# # ViViT Feature Encoder
# # -----------------------
# class ViViTFeat(nn.Module):
#     def __init__(self, feat_dim=2048, hidden_dim=256, num_heads=4, num_layers=2, dropout=0.1):
#         super(ViViTFeat, self).__init__()
#         self.proj = nn.Linear(feat_dim, hidden_dim)
#         self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))

#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=hidden_dim, nhead=num_heads, dropout=dropout, batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
#         self.norm = nn.LayerNorm(hidden_dim)

#     def forward(self, x):
#         """
#         x: [B, T, D] -> img_feat sequence
#         returns: [B, H]  (CLS token)
#         """
#         B, T, D = x.shape
#         x = self.proj(x)  # [B, T, H]
#         cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, H]
#         x = torch.cat((cls_tokens, x), dim=1)          # [B, T+1, H]
#         x = self.transformer(x)                        # [B, T+1, H]
#         return self.norm(x[:, 0])                      # CLS output [B, H]


# # -----------------------
# # Main Model
# # -----------------------
# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dad, self).__init__()

#         self.num_heads = 4
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # -----------------------
#         # Object graph features
#         # -----------------------
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)   # 2048 -> 256
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)         # 300 -> 64
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

#         # -----------------------
#         # Spatial and temporal graph transformers
#         # -----------------------
#         self.gc1_spatial = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,  # 256+64=320
#             out_channels=embedding_dim // 2,                      # 64
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc1_temporal = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph pooling
#         self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

#         # -----------------------
#         # I3D features -> Transformer
#         # -----------------------
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)  # 2048 -> 256
#         encoder_layer = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True
#         )
#         self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=2)

#         # Replace TemporalFusionTransformer with ViViT (CLS)
#         self.vivit_branch = ViViTFeat(
#             feat_dim=img_feat_dim,
#             hidden_dim=embedding_dim * 2,
#             num_heads=4,
#             num_layers=2
#         )

#         # -----------------------
#         # Frame-level graph encoding
#         # -----------------------
#         self.gc2_sg = TransformerConv(
#             in_channels=embedding_dim * self.num_heads,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc2_i3d = TransformerConv(
#             in_channels=embedding_dim * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # -----------------------
#         # Classification
#         # -----------------------
#         concat_dim = (embedding_dim // 2 * self.num_heads) + \
#                      (embedding_dim // 2 * self.num_heads) + \
#                      (embedding_dim * 2)  # includes ViViT CLS contribution per frame
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings,
#                 temporal_adj_list, temporal_edge_w, batch_vec):
#         """
#         x: object features (node-wise)
#         img_feat: I3D features. Expected shape depends on how you feed them:
#                   - If single video without batch: [T, D]  (we will handle by unsqueezing)
#                   - If batched: [B, T, D]
#         video_adj_list: adjacency for frame-level graph (edges connecting frames)
#         """

#         # -----------------------
#         # Object graph processing
#         # -----------------------
#         x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
#         x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
#         x = torch.cat((x_feat, x_label), 1)  # (N, 320)

#         # Spatial graph
#         edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_spatial = self.relu(self.gc1_norm1(
#             self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
#         ))

#         # Temporal graph
#         edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_temporal = self.relu(self.gc1_norm2(
#             self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
#         ))

#         # Concat + pooling
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)  # shape depends on graph pooling: often [num_frames_or_nodes, feat] or [B, feat]

#         # -----------------------
#         # I3D feature processing
#         # -----------------------
#         # Normalize img_feat to [B, T, D] if user passed [T, D]
#         if img_feat.dim() == 2:
#             # single-video case -> make batch dimension
#             img_feat_seq = img_feat.unsqueeze(0)  # [1, T, D]
#             single_video_batch = True
#         else:
#             img_feat_seq = img_feat  # assume already [B, T, D]
#             single_video_batch = False

#         # Project I3D features for the original temporal transformer
#         img_feat_proj = self.img_fc(img_feat_seq)  # [B, T, H]
#         img_feat_orig = self.temporal_transformer(img_feat_proj)  # [B, T, H]

#         # ViViT CLS output: [B, H]
#         vivit_out = self.vivit_branch(img_feat_seq)  # [B, H]

#         # -----------------------
#         # Frame-level embeddings
#         # -----------------------
#         # The graph frame encoders likely expect node-wise inputs.
#         # Depending on how you constructed `video_adj_list` and `g_embed`,
#         # frame-level embeddings may be 2D [N, F] or 3D [B, T, F].
#         frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
#         frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat_orig.view(-1, img_feat_orig.size(-1)), video_adj_list)))
#         # Note: the above view() is defensive — if your gc2_i3d expects [num_nodes, feat],
#         # ensure `video_adj_list` and `g_embed` match that node layout. If your frame graph expects batched frames,
#         # adjust accordingly (e.g., use reshape to [B*T, feat] and remember T).

#         # -----------------------
#         # Robust concatenation: expand vivit_out to match frame embeddings
#         # -----------------------
#         # Handle common cases:
#         # - frame_embed_* are 2D: [N, F]  -> vivit_out should be expanded to [N, H]
#         # - frame_embed_* are 3D: [B, T, F] -> vivit_out should be expanded to [B, T, H]
#         f_sg = frame_embed_sg
#         f_img = frame_embed_img

#         # Ensure dimensions match between frame-level branches
#         # If one of them is 3D and the other 2D, try to make them consistent
#         if f_sg.dim() == 3 and f_img.dim() == 2:
#             # duplicate f_img across time dimension if needed
#             B, T, _ = f_sg.shape
#             f_img = f_img.view(B, 1, -1).expand(-1, T, -1)
#         elif f_sg.dim() == 2 and f_img.dim() == 3:
#             B, T, _ = f_img.shape
#             f_sg = f_sg.view(B, 1, -1).expand(-1, T, -1)

#         # Now expand vivit_out to match
#         if f_sg.dim() == 3:
#             # expected [B, T, H]
#             B, T, _ = f_sg.shape
#             if vivit_out.size(0) == B:
#                 vivit_out_exp = vivit_out.unsqueeze(1).expand(-1, T, -1)  # [B, T, H]
#             elif vivit_out.size(0) == 1:
#                 vivit_out_exp = vivit_out.unsqueeze(1).expand(B, T, -1)
#             else:
#                 # fallback: try to repeat/crop to match B
#                 vivit_out_exp = vivit_out.repeat(int(B / vivit_out.size(0)), 1).unsqueeze(1).expand(-1, T, -1)
#         elif f_sg.dim() == 2:
#             # expected [N, H] where N = num_frames (or nodes)
#             N = f_sg.size(0)
#             if vivit_out.dim() == 2 and vivit_out.size(0) == N:
#                 vivit_out_exp = vivit_out
#             elif vivit_out.dim() == 2 and vivit_out.size(0) == 1:
#                 vivit_out_exp = vivit_out.expand(N, -1)
#             else:
#                 # try to flatten vivit_out and expand/crop
#                 viv_flat = vivit_out.view(-1, vivit_out.size(-1))
#                 vivit_out_exp = viv_flat.expand(N, -1)[:N, :]

#         else:
#             raise RuntimeError("Unexpected frame_embed shape. f_sg.dim() = {}".format(f_sg.dim()))

#         # Finally, concatenate along feature dimension
#         if f_sg.dim() == 3:
#             # [B, T, F1] + [B, T, F2] + [B, T, H] -> concat on dim=2
#             frame_embed_ = torch.cat((f_sg, f_img, vivit_out_exp), dim=2)
#             # If you want to reduce to video-level later, you can mean pool across time here
#         else:
#             # [N, F1] + [N, F2] + [N, H] -> concat on dim=1
#             frame_embed_ = torch.cat((f_sg, f_img, vivit_out_exp), dim=1)

#         # -----------------------
#         # Classification
#         # -----------------------
#         frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
#         logits_mc = self.classify_fc2(frame_embed_)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc

# Transconv - cross spatial and temporal 
import torch
import torch.nn as nn
from torch_geometric.nn import (
    TransformerConv,
    SAGPooling,
    global_max_pool,
    InstanceNorm
)
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class SpaceTempGoG_detr_dota(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(SpaceTempGoG_detr_dota, self).__init__()

        self.num_heads = 4
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim

        # -----------------------
        # Object graph features
        # -----------------------
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

        # -----------------------
        # Spatial and temporal graph transformers
        # -----------------------
        self.gc1_spatial = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        self.gc1_temporal = TransformerConv(
            in_channels=embedding_dim * 2 + embedding_dim // 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads,
            edge_dim=1,
            beta=True
        )
        self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # -----------------------
        # Cross-graph attention
        # -----------------------
        self.gc_cross = TransformerConv(
            in_channels=(embedding_dim // 2 * self.num_heads) * 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads
        )
        self.gc_cross_norm = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # Graph pooling
        self.pool = SAGPooling(embedding_dim // 2 * self.num_heads, ratio=0.8)

        # -----------------------
        # I3D features -> Transformer
        # -----------------------
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim * 2,
            nhead=4,
            batch_first=True
        )
        self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=2)

        encoder_layer_fusion = TransformerEncoderLayer(
            d_model=embedding_dim * 2,
            nhead=4,
            batch_first=True,
            dropout=0.1
        )
        self.temporal_fusion_transformer = TransformerEncoder(encoder_layer_fusion, num_layers=2)

        # -----------------------
        # Frame-level graph encoding
        # -----------------------
        self.gc2_sg = TransformerConv(
            in_channels=embedding_dim // 2 * self.num_heads,
            out_channels=embedding_dim // 2,
            heads=self.num_heads
        )
        self.gc2_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        self.gc2_i3d = TransformerConv(
            in_channels=embedding_dim * 2,
            out_channels=embedding_dim // 2,
            heads=self.num_heads
        )
        self.gc2_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

        # -----------------------
        # Classification
        # -----------------------
        concat_dim = (embedding_dim // 2 * self.num_heads) + \
                     (embedding_dim // 2 * self.num_heads) + \
                     (embedding_dim * 2)
        self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
        self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings,
                temporal_adj_list, temporal_edge_w, batch_vec):

        # -----------------------
        # Object graph processing
        # -----------------------
        x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
        x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
        x = torch.cat((x_feat, x_label), 1)

        # Spatial graph
        edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x.dtype).to(x.device)
        n_embed_spatial = self.relu(self.gc1_norm1(self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)))

        # Temporal graph
        edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x.dtype).to(x.device)
        n_embed_temporal = self.relu(self.gc1_norm2(self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)))

        # -----------------------
        # Cross-graph attention
        # -----------------------
        n_embed_cross = torch.cat((n_embed_spatial, n_embed_temporal), dim=-1)
        n_embed_cross = self.relu(self.gc_cross_norm(self.gc_cross(n_embed_cross, edge_index)))

        # Graph pooling
        n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed_cross, edge_index, None, batch_vec)
        g_embed = global_max_pool(n_embed, batch_vec)

        # -----------------------
        # I3D feature processing
        # -----------------------
        img_feat_orig = self.img_fc(img_feat).unsqueeze(0)
        img_feat_orig = self.temporal_transformer(img_feat_orig, is_causal=True).squeeze(0)

        img_feat_fusion = self.img_fc(img_feat).unsqueeze(0)
        img_feat_fusion = self.temporal_fusion_transformer(img_feat_fusion, is_causal=True).squeeze(0)

        # -----------------------
        # Frame-level embeddings
        # -----------------------
        frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
        frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat_orig, video_adj_list)))

        # Concatenate all features
        frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img, img_feat_fusion), 1)

        # -----------------------
        # Classification
        # -----------------------
        frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
        logits_mc = self.classify_fc2(frame_embed_)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import (
#     TransformerConv,
#     SAGPooling,
#     global_max_pool,
#     InstanceNorm
# )
# from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dad, self).__init__()

#         self.num_heads = 4
#         self.input_dim = input_dim
#         self.embedding_dim = embedding_dim

#         # -----------------------
#         # Object graph features
#         # -----------------------
#         self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
#         self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
#         self.obj_l_fc = nn.Linear(300, embedding_dim // 2)
#         self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

#         # -----------------------
#         # Spatial and temporal graph transformers
#         # -----------------------
#         self.gc1_spatial = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc1_temporal = TransformerConv(
#             in_channels=embedding_dim * 2 + embedding_dim // 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads,
#             edge_dim=1,
#             beta=True
#         )
#         self.gc1_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # Graph pooling
#         self.pool = SAGPooling(embedding_dim * self.num_heads, ratio=0.8)

#         # -----------------------
#         # I3D features -> Transformer
#         # -----------------------
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)
#         encoder_layer = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True
#         )
#         self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=2)

#         # Parallel TemporalFusionTransformer branch
#         encoder_layer_fusion = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True,
#             dropout=0.1
#         )
#         self.temporal_fusion_transformer = TransformerEncoder(encoder_layer_fusion, num_layers=2)

#         # -----------------------
#         # Parallel Decoder branch (context-aware)
#         # -----------------------
#         decoder_layer = TransformerDecoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=4,
#             batch_first=True,
#             dropout=0.1
#         )
#         self.img_feat_decoder = TransformerDecoder(decoder_layer, num_layers=2)
#         self.decoder_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

#         # -----------------------
#         # Frame-level graph encoding
#         # -----------------------
#         self.gc2_sg = TransformerConv(
#             in_channels=embedding_dim * self.num_heads,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm1 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         self.gc2_i3d = TransformerConv(
#             in_channels=embedding_dim * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm2 = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # -----------------------
#         # Classification
#         # -----------------------
#         concat_dim = (embedding_dim // 2 * self.num_heads) + \
#                      (embedding_dim // 2 * self.num_heads) + \
#                      (embedding_dim * 2) + \
#                      (embedding_dim * 2)  # adding decoder branch
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings,
#                 temporal_adj_list, temporal_edge_w, batch_vec):

#         # -----------------------
#         # Object graph processing
#         # -----------------------
#         x_feat = self.relu(self.x_bn1(self.x_fc(x[:, :self.input_dim])))
#         x_label = self.relu(self.obj_l_bn1(self.obj_l_fc(x[:, self.input_dim:])))
#         x = torch.cat((x_feat, x_label), 1)  # (N, 320)

#         # Spatial graph
#         edge_attr_spatial = edge_embeddings[:, -1].unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_spatial = self.relu(self.gc1_norm1(
#             self.gc1_spatial(x, edge_index, edge_attr=edge_attr_spatial)
#         ))

#         # Temporal graph
#         edge_attr_temporal = temporal_edge_w.unsqueeze(1).to(x.dtype).to(x.device)
#         n_embed_temporal = self.relu(self.gc1_norm2(
#             self.gc1_temporal(x, temporal_adj_list, edge_attr=edge_attr_temporal)
#         ))

#         # Concat + pooling
#         n_embed = torch.cat((n_embed_spatial, n_embed_temporal), 1)
#         n_embed, edge_index, _, batch_vec, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
#         g_embed = global_max_pool(n_embed, batch_vec)

#         # -----------------------
#         # I3D feature processing
#         # -----------------------
#         img_feat_orig = self.img_fc(img_feat).unsqueeze(0)
#         img_feat_orig = self.temporal_transformer(img_feat_orig)
#         img_feat_orig = img_feat_orig.squeeze(0)

#         img_feat_fusion = self.img_fc(img_feat).unsqueeze(0)
#         img_feat_fusion = self.temporal_fusion_transformer(img_feat_fusion)
#         img_feat_fusion = img_feat_fusion.squeeze(0)

#         # -----------------------
#         # Parallel Decoder branch (context-aware)
#         # -----------------------
#         decoder_input = self.decoder_fc(img_feat).unsqueeze(0)  # (1, T, D)
#         decoder_out = self.img_feat_decoder(tgt=decoder_input, memory=decoder_input)  # (1, T, D)
#         decoder_out = decoder_out.squeeze(0)  # (T, D) frame-level embeddings

#         # -----------------------
#         # Frame-level graph embeddings
#         # -----------------------
#         frame_embed_sg = self.relu(self.gc2_norm1(self.gc2_sg(g_embed, video_adj_list)))
#         frame_embed_img = self.relu(self.gc2_norm2(self.gc2_i3d(img_feat_orig, video_adj_list)))

#         # -----------------------
#         # Concatenate all features per frame
#         # -----------------------
#         frame_embed_ = torch.cat((frame_embed_sg, frame_embed_img, img_feat_fusion, decoder_out), 1)

#         # -----------------------
#         # Classification
#         # -----------------------
#         frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
#         logits_mc = self.classify_fc2(frame_embed_)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc


# Attention Only but concantenate with itself
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import TransformerConv, InstanceNorm
# from torch.nn import MultiheadAttention

# class SpaceTempGoG_detr_dota(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dota, self).__init__()

#         self.embedding_dim = embedding_dim
#         self.num_heads = 4

#         # -----------------------
#         # Image feature projection
#         # -----------------------
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

#         # -----------------------
#         # Multihead attention branch (self-attention)
#         # -----------------------
#         self.img_attn = MultiheadAttention(
#             embed_dim=embedding_dim * 2,
#             num_heads=self.num_heads,
#             batch_first=True
#         )

#         # -----------------------
#         # Optional fusion layer
#         # -----------------------
#         self.fusion_fc = nn.Linear(embedding_dim * 2, embedding_dim * 2)

#         # -----------------------
#         # Single Graph TransformerConv branch
#         # -----------------------
#         self.gc = TransformerConv(
#             in_channels=embedding_dim * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.norm = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # -----------------------
#         # Classification
#         # -----------------------
#         concat_dim = (embedding_dim // 2 * self.num_heads) + embedding_dim * 2  # single graph + attention features
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings=None,
#                 temporal_adj_list=None, temporal_edge_w=None, batch_vec=None):
#         """
#         img_feat: (seq_len, img_feat_dim)
#         video_adj_list: graph edges for TransformerConv
#         """

#         # -----------------------
#         # Image feature projection
#         # -----------------------
#         img_feat_proj = self.img_fc(img_feat)  # (seq_len, d_model)

#         # -----------------------
#         # Multihead attention
#         # -----------------------
#         img_feat_attn, _ = self.img_attn(
#             img_feat_proj.unsqueeze(0),  # Q
#             img_feat_proj.unsqueeze(0),  # K
#             img_feat_proj.unsqueeze(0),  # V
#             is_causal=True
#         )
#         img_feat_attn = img_feat_attn.squeeze(0)

#         # -----------------------
#         # Fusion layer
#         # -----------------------
#         img_feat_fused = self.fusion_fc(img_feat_attn)

#         # -----------------------
#         # Single Graph TransformerConv
#         # -----------------------
#         frame_embed = self.relu(self.norm(self.gc(img_feat_fused, video_adj_list)))

#         # -----------------------
#         # Concatenate features
#         # -----------------------
#         frame_embed_ = torch.cat((frame_embed, img_feat_fused), dim=1)

#         # -----------------------
#         # Classification
#         # -----------------------
#         frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
#         logits = self.classify_fc2(frame_embed_)
#         probs = self.softmax(logits)

#         return logits, probs


# # STAGNet Image Only part - STAGNet Lite
import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv, InstanceNorm

class SpaceTempGoG_detr_dota(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
        super(SpaceTempGoG_detr_dota, self).__init__()
        self.num_heads = 1
        self.embedding_dim = embedding_dim

        # I3D features with temporal processing
        self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

        # LSTM for temporal sequence processing
        self.temporal_lstm = nn.LSTM(
            input_size=embedding_dim * 2,
            hidden_size=embedding_dim * 2,
            num_layers=1,
            batch_first=True
        )

        # Frame-level graph convolution using only I3D features
        self.gc2_i3d = GATv2Conv(
            embedding_dim * 2,  # Input from LSTM output
            embedding_dim // 2, 
            heads=self.num_heads
        )
        self.gc2_norm2 = InstanceNorm(embedding_dim // 2)

        # Classifier
        self.classify_fc1 = nn.Linear(embedding_dim // 2, embedding_dim // 2)
        self.classify_fc2 = nn.Linear(embedding_dim // 2, num_classes)

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings=None,
                temporal_adj_list=None, temporal_edge_w=None, batch_vec=None):
        # Process I3D feature
        img_feat = self.img_fc(img_feat)

        # LSTM temporal modeling
        img_feat = img_feat.unsqueeze(0)  # (1, seq_len, feat_dim)
        img_feat, (_, _) = self.temporal_lstm(img_feat)
        img_feat = img_feat.squeeze(0)  # (seq_len, feat_dim)

        # Frame-level graph convolution
        frame_embed_img = self.relu(
            self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list))
        )

        # Classification
        frame_embed_img = self.relu(self.classify_fc1(frame_embed_img))
        logits_mc = self.classify_fc2(frame_embed_img)
        probs_mc = self.softmax(logits_mc)

        return logits_mc, probs_mc

# 1 Transformer Only
# import torch
# import torch.nn as nn
# from torch_geometric.nn import TransformerConv, InstanceNorm
# from torch.nn import TransformerEncoder, TransformerEncoderLayer

# class SpaceTempGoG_detr_dota(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dota, self).__init__()
#         self.num_heads = 1
#         self.embedding_dim = embedding_dim

#         # I3D feature projection
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

#         # Temporal Transformer Encoder
#         encoder_layer = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=self.num_heads,
#             batch_first=True
#         )
#         self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=1)

#         # Frame-level graph convolution using only transformed features
#         self.gc2_i3d = TransformerConv(
#             embedding_dim * 2,
#             embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm2 = InstanceNorm(embedding_dim // 2)

#         # Classifier
#         self.classify_fc1 = nn.Linear(embedding_dim // 2, embedding_dim // 2)
#         self.classify_fc2 = nn.Linear(embedding_dim // 2, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings=None,
#                 temporal_adj_list=None, temporal_edge_w=None, batch_vec=None):
#         # Project I3D features
#         img_feat = self.img_fc(img_feat)

#         # Temporal Transformer modeling
#         img_feat = img_feat.unsqueeze(0)  # (1, seq_len, feat_dim)
#         img_feat = self.temporal_transformer(img_feat, is_causal=True)
#         img_feat = img_feat.squeeze(0)  # (seq_len, feat_dim)

#         # Frame-level graph convolution
#         frame_embed_img = self.relu(
#             self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list))
#         )

#         # Classification
#         frame_embed_img = self.relu(self.classify_fc1(frame_embed_img))
#         logits_mc = self.classify_fc2(frame_embed_img)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc


# Multi-head only
# import torch
# import torch.nn as nn
# from torch_geometric.nn import TransformerConv, InstanceNorm
# from torch.nn import MultiheadAttention

# class SpaceTempGoG_detr_dota(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dota, self).__init__()
#         self.num_heads = 1
#         self.embedding_dim = embedding_dim

#         # I3D feature projection
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

#         # Temporal Multihead Attention (replaces TransformerEncoder)
#         self.img_attn = MultiheadAttention(
#             embed_dim=embedding_dim * 2,
#             num_heads=self.num_heads,
#             batch_first=True
#         )

#         # Frame-level graph convolution using only transformed features
#         self.gc2_i3d = TransformerConv(
#             embedding_dim * 2,
#             embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc2_norm2 = InstanceNorm(embedding_dim // 2)

#         # Classifier
#         self.classify_fc1 = nn.Linear(embedding_dim // 2, embedding_dim // 2)
#         self.classify_fc2 = nn.Linear(embedding_dim // 2, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings=None,
#                 temporal_adj_list=None, temporal_edge_w=None, batch_vec=None):
#         # Project I3D features
#         img_feat = self.img_fc(img_feat)

#         # -------------------------
#         # Temporal modeling block
#         # -------------------------
#         img_feat_trans = img_feat  # projected features
#         img_feat_attn, _ = self.img_attn(
#             img_feat_trans.unsqueeze(0),  # Q
#             img_feat_trans.unsqueeze(0),  # K
#             img_feat_trans.unsqueeze(0),  # V
#             is_causal=True
#         )
#         img_feat = img_feat_attn.squeeze(0)  # (seq_len, feat_dim)

#         # Frame-level graph convolution
#         frame_embed_img = self.relu(
#             self.gc2_norm2(self.gc2_i3d(img_feat, video_adj_list))
#         )

#         # Classification
#         frame_embed_img = self.relu(self.classify_fc1(frame_embed_img))
#         logits_mc = self.classify_fc2(frame_embed_img)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc



# Combined - 1 Transformer 1 Multi-head
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import TransformerConv, InstanceNorm
# from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention

# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dad, self).__init__()

#         self.embedding_dim = embedding_dim
#         self.num_heads = 1

#         # -----------------------
#         # Image feature projection
#         # -----------------------
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

#         # -----------------------
#         # Temporal TransformerEncoder
#         # -----------------------
#         encoder_layer = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=self.num_heads,
#             batch_first=True
#         )
#         self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=2)

#         # -----------------------
#         # Multihead attention branch
#         # -----------------------
#         self.img_attn = MultiheadAttention(
#             embed_dim=embedding_dim * 2,
#             num_heads=self.num_heads,
#             batch_first=True
#         )

#         # -----------------------
#         # Graph TransformerConv branches
#         # -----------------------
#         self.gc_tran = TransformerConv(
#             in_channels=embedding_dim * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc_attn = TransformerConv(
#             in_channels=embedding_dim * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.norm_tran = InstanceNorm(embedding_dim // 2 * self.num_heads)
#         self.norm_attn = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # -----------------------
#         # Classification
#         # -----------------------
#         concat_dim = (embedding_dim // 2 * self.num_heads) * 2
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings=None,
#                 temporal_adj_list=None, temporal_edge_w=None, batch_vec=None):

#         # -----------------------
#         # Image feature projection (overwrite img_feat)
#         # -----------------------
#         img_feat = self.img_fc(img_feat)  # (seq_len, d_model)

#         # -----------------------
#         # Temporal Transformer
#         # -----------------------
#         img_feat_trans = img_feat.unsqueeze(0)  # (1, seq_len, d_model)
#         img_feat_tran = self.temporal_transformer(img_feat_trans, is_causal=True)
#         img_feat_tran = img_feat_tran.squeeze(0)  # (seq_len, d_model)

#         # -----------------------
#         # Multihead attention using original img_feat
#         # -----------------------
#         img_feat_attn, _ = self.img_attn(
#             img_feat.unsqueeze(0),  # Q
#             img_feat.unsqueeze(0),  # K
#             img_feat.unsqueeze(0),  # V
#             is_causal=True
#         )
#         img_feat_attn = img_feat_attn.squeeze(0)

#         # -----------------------
#         # Graph TransformerConv
#         # -----------------------
#         frame_embed_tran = self.relu(self.norm_tran(self.gc_tran(img_feat_tran, video_adj_list)))
#         frame_embed_attn = self.relu(self.norm_attn(self.gc_attn(img_feat_attn, video_adj_list)))

#         # -----------------------
#         # Concatenate all features
#         # -----------------------
#         frame_embed_img = torch.cat((frame_embed_tran, frame_embed_attn), dim=1)

#         # -----------------------
#         # Classification
#         # -----------------------
#         frame_embed_img = self.relu(self.classify_fc1(frame_embed_img))
#         logits_mc = self.classify_fc2(frame_embed_img)
#         probs_mc = self.softmax(logits_mc)

#         return logits_mc, probs_mc


## Trans then multi then combine three output
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import TransformerConv, InstanceNorm
# from torch.nn import TransformerEncoder, TransformerEncoderLayer, MultiheadAttention

# class SpaceTempGoG_detr_dad(nn.Module):
#     def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2):
#         super(SpaceTempGoG_detr_dad, self).__init__()

#         self.embedding_dim = embedding_dim
#         self.num_heads = 4

#         # -----------------------
#         # Image feature projection
#         # -----------------------
#         self.img_fc = nn.Linear(img_feat_dim, embedding_dim * 2)

#         # -----------------------
#         # Single causal TransformerEncoder
#         # -----------------------
#         encoder_layer = TransformerEncoderLayer(
#             d_model=embedding_dim * 2,
#             nhead=self.num_heads,
#             batch_first=True
#         )
#         self.temporal_transformer = TransformerEncoder(encoder_layer, num_layers=2)

#         # -----------------------
#         # Optional fusion layer to capture complementary patterns
#         # -----------------------
#         self.fusion_fc = nn.Linear(embedding_dim * 2, embedding_dim * 2)

#         # -----------------------
#         # Multihead attention branch (self-attention)
#         # -----------------------
#         self.img_attn = MultiheadAttention(
#             embed_dim=embedding_dim * 2,
#             num_heads=self.num_heads,
#             batch_first=True
#         )

#         # -----------------------
#         # Graph TransformerConv branches
#         # -----------------------
#         self.gc_orig = TransformerConv(
#             in_channels=embedding_dim * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.gc_attn = TransformerConv(
#             in_channels=embedding_dim * 2,
#             out_channels=embedding_dim // 2,
#             heads=self.num_heads
#         )
#         self.norm_orig = InstanceNorm(embedding_dim // 2 * self.num_heads)
#         self.norm_attn = InstanceNorm(embedding_dim // 2 * self.num_heads)

#         # -----------------------
#         # Classification
#         # -----------------------
#         concat_dim = (embedding_dim // 2 * self.num_heads) * 2 + embedding_dim * 2
#         self.classify_fc1 = nn.Linear(concat_dim, embedding_dim)
#         self.classify_fc2 = nn.Linear(embedding_dim, num_classes)

#         self.relu = nn.LeakyReLU(0.2)
#         self.softmax = nn.Softmax(dim=-1)

#     def forward(self, x, edge_index, img_feat, video_adj_list, edge_embeddings=None,
#                 temporal_adj_list=None, temporal_edge_w=None, batch_vec=None):
#         """
#         img_feat: (seq_len, img_feat_dim)
#         video_adj_list: graph edges for TransformerConv
#         """

#         # -----------------------
#         # Image feature projection
#         # -----------------------
#         img_feat_proj = self.img_fc(img_feat).unsqueeze(0)  # (1, seq_len, d_model)

#         # -----------------------
#         # Single causal Transformer
#         # -----------------------
#         img_feat_trans = self.temporal_transformer(img_feat_proj, is_causal=True)
#         img_feat_trans = self.fusion_fc(img_feat_trans.squeeze(0))  # fusion after transformer

#         # -----------------------
#         # Multihead attention branch
#         # -----------------------
#         img_feat_attn, _ = self.img_attn(
#             img_feat_trans.unsqueeze(0),  # Q
#             img_feat_trans.unsqueeze(0),  # K
#             img_feat_trans.unsqueeze(0),  # V
#             is_causal=True
#         )
#         img_feat_attn = img_feat_attn.squeeze(0)

#         # -----------------------
#         # Graph TransformerConv
#         # -----------------------
#         frame_embed_orig = self.relu(self.norm_orig(self.gc_orig(img_feat_trans, video_adj_list)))
#         frame_embed_attn = self.relu(self.norm_attn(self.gc_attn(img_feat_attn, video_adj_list)))

#         # -----------------------
#         # Concatenate all features
#         # -----------------------
#         frame_embed_ = torch.cat((frame_embed_orig, frame_embed_attn, img_feat_trans), dim=1)

#         # -----------------------
#         # Classification
#         # -----------------------
#         frame_embed_ = self.relu(self.classify_fc1(frame_embed_))
#         logits = self.classify_fc2(frame_embed_)
#         probs = self.softmax(logits)

#         return logits, probs










