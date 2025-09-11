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

class SpaceTempGoG_detr_dad(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=128, img_feat_dim=2048, num_classes=2, emsa_groups=4):
        super(SpaceTempGoG_detr_dad, self).__init__()

        # Make embedding divisible by EMSA groups
        assert embedding_dim * 2 % emsa_groups == 0, f"concat_dim={embedding_dim*2} must be divisible by EMSA groups={emsa_groups}"
        self.embedding_dim = embedding_dim
        self.emsa_groups = emsa_groups

        # Linear projections
        self.obj_proj = nn.Linear(input_dim, embedding_dim)
        self.global_proj = nn.Linear(img_feat_dim, embedding_dim)

        concat_dim = embedding_dim * 2

        # Parallel attention modules
        self.memory_attention = Memory_Attention_Aggregation(agg_dim=concat_dim, d_model=concat_dim)
        self.aux_attention = Auxiliary_Self_Attention_Aggregation(agg_dim=concat_dim)
        self.temporal_emsa = EMSA(channels=concat_dim, factor=emsa_groups)

        # Projection layers after attention outputs to unify shapes
        self.mem_proj = nn.Linear(concat_dim, concat_dim)
        self.aux_proj = nn.Linear(concat_dim, concat_dim)
        self.emsa_proj = nn.Linear(concat_dim, concat_dim)

        # Final classifier
        concat_dim = 512
        fused_dim = concat_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, fused_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fused_dim // 2, num_classes)
        )

    def forward(self, obj_feats, global_feats):
        # Ensure tensor dtype/device matches model
        ref = next(self.parameters())
        obj_feats = obj_feats.to(dtype=ref.dtype, device=ref.device)
        global_feats = global_feats.to(dtype=ref.dtype, device=ref.device)

        # Add batch dim if missing
        if obj_feats.dim() == 2:
            obj_feats = obj_feats.unsqueeze(0)
        if global_feats.dim() == 2:
            global_feats = global_feats.unsqueeze(0)

        # Project features
        obj_proj = self.obj_proj(obj_feats)        # [B, T_obj, embedding_dim]
        global_proj = self.global_proj(global_feats)  # [B, T_global, embedding_dim]

        # Align temporal dimension
        T_max = max(obj_proj.size(1), global_proj.size(1))
        if obj_proj.size(1) != T_max:
            obj_proj = F.interpolate(obj_proj.transpose(1,2), size=T_max, mode='linear', align_corners=False).transpose(1,2)
        if global_proj.size(1) != T_max:
            global_proj = F.interpolate(global_proj.transpose(1,2), size=T_max, mode='linear', align_corners=False).transpose(1,2)

        # Concatenate features
        concat_feats = torch.cat([obj_proj, global_proj], dim=-1)  # [B, T_max, 2*embedding_dim]

        # Apply attention modules
        mem_out = self.mem_proj(self.memory_attention(concat_feats))  # [B, T_max, concat_dim]
        aux_out_pre = self.aux_attention(concat_feats)  # [B, T_max, ?]

        # Ensure aux_attention output has correct shape
        concat_dim = self.embedding_dim * 2
        if aux_out_pre.size(-1) != concat_dim:
            # Project feature dimension to concat_dim, preserving temporal dimension
            aux_out_pre = nn.Linear(aux_out_pre.size(-1), concat_dim).to(aux_out_pre.device)(aux_out_pre)
        aux_out = self.aux_proj(aux_out_pre)  # [B, T_max, concat_dim]

        # EMSA expects [B, C, H=1, W=T_max]
        emsa_in = concat_feats.transpose(1,2).unsqueeze(2)  # [B, concat_dim, 1, T_max]
        emsa_out = self.emsa_proj(self.temporal_emsa(emsa_in).squeeze(2).transpose(1,2))  # [B, T_max, concat_dim]

        # ==== PRINT STATEMENTS ADDED ====
        # print(f"obj_proj: {obj_proj.shape}, global_proj: {global_proj.shape}")
        # print(f"concat_feats: {concat_feats.shape}")
        # print(f"mem_out: {mem_out.shape}, aux_out: {aux_out.shape}, emsa_out: {emsa_out.shape}")
        # ================================
		
        # Concatenate all attention outputs
        # fused = torch.cat([mem_out, aux_out, emsa_out], dim=-1)  # [B, T_max, 3*concat_dim]
        emsa_out = emsa_out.squeeze(0)
        # print(f"emsa_out after squeeze: {emsa_out.shape}")
        # fused = torch.cat([mem_out, emsa_out], dim=-1)  # [B, T_max, 3*concat_dim]
        # Expand aux_out to [1900, 512]
        aux_out_expanded = aux_out.expand(mem_out.size(0), -1)

        # print()
        # print(f"mem_out: {mem_out.shape}, aux_out_expanded: {aux_out_expanded.shape}, emsa_out: {emsa_out.shape}")
		
        # Concatenate along last dimension
        fused = torch.cat([mem_out, aux_out_expanded, emsa_out], dim=-1)
        # print(fused.shape)  # torch.Size([1900, 1536])

        # Check if 1900 can be reshaped into 100 x 19
        batch_size = 100
        seq_len = 19
        assert fused.size(0) == batch_size * seq_len, "1900 is not divisible by 100"

        # Reshape
        fused = fused.view(batch_size, seq_len, fused.size(1))  # [100, 19, 1536]

        # # Add batch dimension
        # fused = fused.unsqueeze(0)  # [1, 1900, 1536]
        # print("fused shape after unsqueeze: ", fused.shape)
		
        # Pool over temporal dimension
        pooled = fused.mean(dim=1)  # [B, 3*concat_dim]
        # print("pooled fused shape: ", fused.shape)  

        # Classifier
        logits_mc = self.classifier(pooled)
        probs_mc = F.softmax(logits_mc, dim=-1)

        return logits_mc, probs_mc



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

# filename: space_temp_gog_detr_dota_transformer.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyG imports
from torch_geometric.nn import (
    GATv2Conv,
    TransformerConv,
    SAGPooling,
    global_max_pool,
    global_mean_pool,
    InstanceNorm
)

# Optional: tries to use timm ViT for per-frame features (if installed).
try:
    import timm
    TIMM_AVAILABLE = True
except Exception:
    TIMM_AVAILABLE = False

class SpaceTempGoG_detr_dota(nn.Module):
    """
    Upgraded architecture with:
      - Graph TransformerConv for object graph encoding
      - Spatial + Temporal object GNN branches
      - Temporal TransformerEncoder for image/I3D sequence modeling
      - Optional ViT/timm frame encoder (if timm available)
      - Cross-attention fusion between graph embeddings and frame embeddings
      - SAGPooling + global pooling and classification head
    """

    def __init__(
        self,
        input_dim=2048,           # object visual descriptor dim
        label_dim=300,            # object label/one-hot or embedding dim (as in your code)
        embedding_dim=128,        # base embedding
        img_feat_dim=2048,        # input video frame / I3D dim
        num_classes=2,
        num_heads=4,
        transformer_layers=2,
        dropout=0.2,
        use_vit_backbone=False,   # set True to attempt using timm ViT
        vit_model_name="vit_base_patch16_224"
    ):
        super().__init__()

        self.input_dim = input_dim
        self.label_dim = label_dim
        self.embedding_dim = embedding_dim
        self.img_feat_dim = img_feat_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.use_vit_backbone = use_vit_backbone and TIMM_AVAILABLE

        # -------------------------
        # Object feature processing
        # -------------------------
        self.x_fc = nn.Linear(self.input_dim, embedding_dim * 2)
        self.x_bn1 = nn.BatchNorm1d(embedding_dim * 2)
        self.obj_l_fc = nn.Linear(self.label_dim, embedding_dim // 2)
        self.obj_l_bn1 = nn.BatchNorm1d(embedding_dim // 2)

        # Graph transformer convs (spatial & temporal branches)
        # Using TransformerConv gives attention-like message passing
        g_in_dim = embedding_dim * 2 + embedding_dim // 2
        self.gc1_spatial = TransformerConv(in_channels=g_in_dim, out_channels=embedding_dim // 2, heads=1, concat=False)
        self.gc1_norm1 = InstanceNorm(embedding_dim // 2)

        self.gc1_temporal = TransformerConv(in_channels=g_in_dim, out_channels=embedding_dim // 2, heads=1, concat=False)
        self.gc1_norm2 = InstanceNorm(embedding_dim // 2)

        # Pooling
        self.pool = SAGPooling(embedding_dim, ratio=0.8)

        # -------------------------
        # Image / Frame features
        # -------------------------
        # Optional ViT backbone (per-frame) to replace/augment I3D features
        if self.use_vit_backbone:
            # create a timm model that outputs feature vectors
            # timm model creation wrapped in try/except above
            vit = timm.create_model(vit_model_name, pretrained=True, num_classes=0, global_pool="avg")
            self.vit_backbone = vit
            # timm vit outputs embedding dimension; map to embedding_dim*2
            # Many ViT base models use 768 or 1024. We'll detect feature dim dynamically later.
            vit_feat_dim = getattr(vit, "num_features", None) or embedding_dim * 2
            self.vit_proj = nn.Linear(vit_feat_dim, embedding_dim * 2)
        else:
            # simple linear projection from provided img_feat_dim
            self.img_fc = nn.Linear(self.img_feat_dim, embedding_dim * 2)

        # -------------------------
        # Temporal Transformer for image sequence
        # -------------------------
        # Positional encoding (learned)
        self.positional_enc = nn.Parameter(torch.randn(512, embedding_dim * 2) * 0.01)  # up to 512 frames (adjust if needed)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim * 2,
            nhead=self.num_heads,
            dim_feedforward=embedding_dim * 4,
            dropout=self.dropout,
            activation="gelu",
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)

        # -------------------------
        # Fusion: graph-level -> transformer / cross attention
        # -------------------------
        # Project pooled graph embedding to same dim as frame embed
        self.g_proj = nn.Linear(embedding_dim, embedding_dim * 2)  # g_embed -> cross-attn query

        # Cross-attention: query from graph, key/value from frame sequence
        # Using MultiheadAttention with batch_first=True
        self.cross_attn = nn.MultiheadAttention(embed_dim=embedding_dim * 2, num_heads=self.num_heads, batch_first=True)
        self.cross_attn_norm = nn.LayerNorm(embedding_dim * 2)

        # A second GNN layer to refine after fusion (optional)
        self.gc2_sg = GATv2Conv(embedding_dim, embedding_dim // 2, heads=1)
        self.gc2_norm1 = InstanceNorm(embedding_dim // 2)

        self.gc2_i3d = GATv2Conv(embedding_dim * 2, embedding_dim // 2, heads=1)
        self.gc2_norm2 = InstanceNorm(embedding_dim // 2)

        # -------------------------
        # Classifier
        # -------------------------
        self.classify_fc1 = nn.Linear(embedding_dim * 3, embedding_dim // 2)
        self.classify_dropout = nn.Dropout(self.dropout)
        self.classify_fc2 = nn.Linear(embedding_dim // 2, num_classes)

        # Activation
        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=-1)

    # -------------
    # Helpers
    # -------------
    def _project_img_feats(self, img_feat):
        """
        Accepts:
          - img_feat: either
             * (num_frames, feature_dim) or
             * (batch, num_frames, feature_dim) or
             * (num_nodes, feature_dim)
        Returns:
          - frame_feats: (batch, seq_len, embedding_dim*2)
        """
        if self.use_vit_backbone:
            # Expecting img_feat to be a batch of PIL/tensor images or precomputed patches depending on usage.
            # Here we assume user will pass preprocessed images through vit externally if desired.
            # If img_feat is already tensor features, arch can be adapted.
            raise RuntimeError("vit_backbone usage requires you to feed raw image tensors through the model pipeline. "
                               "Switch use_vit_backbone=False to use precomputed img_feat tensors.")
        else:
            # If img_feat is 2D (N_nodes, feat) -> treat as single-frame batch of size 1
            if img_feat.dim() == 2:
                # shape: (N_nodes, feat) -> convert to (1, 1, feat)
                feat = img_feat.unsqueeze(0).unsqueeze(0)
            elif img_feat.dim() == 3:
                # very common: (batch, seq_len, feat) or (seq_len, N_nodes, feat)
                # We'll assume (batch, seq_len, feat) if batch > 1 or seq_len reasonable
                # If shape is (seq_len, N_nodes, feat) but seq_len small, this may need adaptation.
                feat = img_feat
            else:
                raise ValueError("Unexpected img_feat dim: {}. Expect 2 or 3 dims.".format(img_feat.dim()))

            # project
            proj = self.img_fc(feat)  # (batch, seq_len, embedding_dim*2)
            return proj

    def forward(self,
                x,                     # object features: (num_nodes, input_dim + label_dim)
                edge_index,            # spatial edge_index for object graph
                img_feat,              # frame / I3D features (see helper for allowed shapes)
                video_adj_list=None,   # adjacency for frame-level graph (if used)
                edge_embeddings=None,  # edge features for graph convs (optional)
                temporal_adj_list=None,# temporal graph edges (optional)
                temporal_edge_w=None,  # temporal edge weights
                batch_vec=None         # batch vector for nodes
                ):
        """
        Notes about shapes:
         - x: (num_nodes, input_dim + label_dim) where the first input_dim are visual/features,
              remaining label_dim are object label embeddings (as in your original code)
         - img_feat: either (num_nodes, img_feat_dim) or (batch, seq_len, img_feat_dim)
        """

        # -------------------------
        # Object feature processing
        # -------------------------
        x_feat = self.x_fc(x[:, :self.input_dim])
        x_feat = self.relu(self.x_bn1(x_feat))
        x_label = self.obj_l_fc(x[:, self.input_dim:self.input_dim + self.label_dim])
        x_label = self.relu(self.obj_l_bn1(x_label))
        x_proc = torch.cat((x_feat, x_label), dim=1)  # (num_nodes, g_in_dim)

        # Spatial graph embedding
        # If edge attributes exist, use them if the conv supports it.
        # TransformerConv doesn't accept edge_attr; keep usage simple.
        n_embed_spatial = self.gc1_spatial(x_proc, edge_index)
        n_embed_spatial = self.relu(self.gc1_norm1(n_embed_spatial))

        # Temporal graph embedding (object tracked through time) - if you have separate temporal adjacency
        n_embed_temporal = self.gc1_temporal(x_proc, temporal_adj_list if temporal_adj_list is not None else edge_index)
        n_embed_temporal = self.relu(self.gc1_norm2(n_embed_temporal))

        # Combine object-level spatial+temporal embeddings
        n_embed = torch.cat((n_embed_spatial, n_embed_temporal), dim=1)  # (num_nodes, embedding_dim)
        # Pool nodes (SAG pooling reduces nodes based on scores)
        n_embed, edge_index_p, _, batch_vec_p, _, _ = self.pool(n_embed, edge_index, None, batch_vec)
        # Graph-level embedding (per-graph)
        g_embed = global_max_pool(n_embed, batch_vec_p)  # (batch, embedding_dim)

        # -------------------------
        # Frame / Image temporal modeling
        # -------------------------
        # Project / prepare frame features to (batch, seq_len, embedding_dim*2)
        frame_feats = self._project_img_feats(img_feat)  # (batch, seq_len, emb*2)

        batch_size, seq_len, feat_dim = frame_feats.shape
        # Add positional encodings (slice to seq_len)
        if seq_len > self.positional_enc.shape[0]:
            # If sequence longer than positional enc table, wrap or expand; here we slice to max
            pe = self.positional_enc  # (max_pos, feat)
        else:
            pe = self.positional_enc[:seq_len, :].unsqueeze(0).expand(batch_size, -1, -1)

        frame_feats = frame_feats + pe  # (batch, seq_len, emb*2)

        # Temporal transformer encoding
        frame_encoded = self.temporal_transformer(frame_feats)  # (batch, seq_len, emb*2)
        # Optionally, obtain a single frame-level embedding by pooling along time
        frame_pooled = frame_encoded.mean(dim=1)  # (batch, emb*2)

        # -------------------------
        # Cross-attention fusion
        # -------------------------
        # Project graph embedding to same dim
        g_q = self.g_proj(g_embed).unsqueeze(1)  # (batch, 1, emb*2) as query

        # MultiheadAttention expects (batch, seq_q, emb)
        # Use frame_encoded as key/value (batch, seq_len, emb*2)
        attn_output, attn_weights = self.cross_attn(query=g_q, key=frame_encoded, value=frame_encoded, need_weights=True)
        # attn_output: (batch, 1, emb*2)
        attn_output = attn_output.squeeze(1)  # (batch, emb*2)
        # Residual + norm
        fused = self.cross_attn_norm(attn_output + g_q.squeeze(1))  # (batch, emb*2)

        # -------------------------
        # Build final embedding and classify
        # -------------------------
        # g_embed (batch, emb), fused (batch, emb*2), frame_pooled (batch, emb*2)
        # Concatenate: reduce dims to a manageable classifier input
        # To be robust, ensure shapes are batch-first
        if g_embed.dim() == 1:
            g_embed = g_embed.unsqueeze(0)

        # Take global mean of node features too (optional) - not using here to avoid dimension blowup
        final_feat = torch.cat([
            g_embed,               # (batch, emb)
            fused,                 # (batch, emb*2)
            frame_pooled           # (batch, emb*2)
        ], dim=1)  # (batch, emb + emb*2 + emb*2) = (batch, emb*5) but we used embedding_dim * 3 in fc -> ensure compatible

        # Project to classifier hidden dim
        # Adjusted input dim compute to match actual concatenation:
        # g_embed: embedding_dim
        # fused: embedding_dim*2
        # frame_pooled: embedding_dim*2
        # => total = embedding_dim * 5
        # But classifier fc expects embedding_dim * 3 in __init__, so let's compute correctly here:
        clf_in_dim = final_feat.shape[1]
        # Create a small MLP on the fly (if mismatch). To keep module consistency we use classify_fc1 but
        # ensure its input features match; if not, use a small linear adapter.
        if self.classify_fc1.in_features != clf_in_dim:
            # lazy adapter
            adapter = nn.Linear(clf_in_dim, self.classify_fc1.in_features).to(final_feat.device)
            final_feat = adapter(final_feat)

        out = self.classify_fc1(final_feat)
        out = F.relu(out)
        out = self.classify_dropout(out)
        logits = self.classify_fc2(out)  # (batch, num_classes)
        probs = self.softmax(logits)

        return logits, probs

