__all__ = ['PatchTST']

# Cell
from typing import Callable, Optional
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F
import numpy as np

from layers.PatchTST_backbone import PatchTST_backbone
from layers.PatchTST_backbone_feature import PatchTST_backbone_feature
from layers.PatchTST_layers import series_decomp



class AutocorrelatedTower(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size, embedding_dim, configs,max_seq_len:Optional[int]=1024,d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs
):
        super(AutocorrelatedTower, self).__init__()
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        self.embedding = nn.Embedding(input_size, embedding_dim)

        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone(c_in=c_in, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)

        self.output_layer = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        # x = self.embedding(x)
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        # return x
        # print('transformed_output2',transformed_output.shape)
        # transformed_output = torch.mean(transformed_output, dim=1)
        # print('transformed_output3',transformed_output.shape)
        # final_output = self.output_layer(transformed_output)
        # x = torch.mean(x, dim=1)  # 沿着序列长度取平均
        # x = x.permute(0,2,1) 
        # print('x',x.shape)
        # final_output = self.output_layer(x)
        # print('final_output',final_output.shape)
        # return final_output
        return x

class featureTower(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads, num_layers, output_size, embedding_dim, configs,max_seq_len:Optional[int]=1024,d_k:Optional[int]=None, d_v:Optional[int]=None, norm:str='BatchNorm', attn_dropout:float=0., 
                 act:str="gelu", key_padding_mask:bool='auto',padding_var:Optional[int]=None, attn_mask:Optional[Tensor]=None, res_attention:bool=True, 
                 pre_norm:bool=False, store_attn:bool=False, pe:str='zeros', learn_pe:bool=True, pretrain_head:bool=False, head_type = 'flatten', verbose:bool=False, **kwargs
):
        super(featureTower, self).__init__()
        # load parameters
        c_in = configs.enc_in
        context_window = configs.seq_len
        target_window = configs.pred_len
        
        n_layers = configs.e_layers
        n_heads = configs.n_heads
        d_model = configs.d_model
        d_ff = configs.d_ff
        dropout = configs.dropout
        fc_dropout = configs.fc_dropout
        head_dropout = configs.head_dropout
        
        individual = configs.individual
    
        patch_len = configs.patch_len
        stride = configs.stride
        padding_patch = configs.padding_patch
        
        revin = configs.revin
        affine = configs.affine
        subtract_last = configs.subtract_last
        
        decomposition = configs.decomposition
        kernel_size = configs.kernel_size

        self.embedding = nn.Embedding(input_size, embedding_dim)

        self.decomposition = decomposition
        if self.decomposition:
            self.decomp_module = series_decomp(kernel_size)
            self.model_trend = PatchTST_backbone_feature(c_in=c_in,configs=configs, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
            self.model_res = PatchTST_backbone_feature(c_in=c_in,configs=configs, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)
        else:
            self.model = PatchTST_backbone_feature(c_in=c_in,configs=configs, context_window = context_window, target_window=target_window, patch_len=patch_len, stride=stride, 
                                  max_seq_len=max_seq_len, n_layers=n_layers, d_model=d_model,
                                  n_heads=n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, norm=norm, attn_dropout=attn_dropout,
                                  dropout=dropout, act=act, key_padding_mask=key_padding_mask, padding_var=padding_var, 
                                  attn_mask=attn_mask, res_attention=res_attention, pre_norm=pre_norm, store_attn=store_attn,
                                  pe=pe, learn_pe=learn_pe, fc_dropout=fc_dropout, head_dropout=head_dropout, padding_patch = padding_patch,
                                  pretrain_head=pretrain_head, head_type=head_type, individual=individual, revin=revin, affine=affine,
                                  subtract_last=subtract_last, verbose=verbose, **kwargs)

        self.output_layer = nn.Linear(input_size, output_size)
    
    def forward(self, x):
        # x = self.embedding(x)
        if self.decomposition:
            res_init, trend_init = self.decomp_module(x)
            res_init, trend_init = res_init.permute(0,2,1), trend_init.permute(0,2,1)  # x: [Batch, Channel, Input length]
            res = self.model_res(res_init)
            trend = self.model_trend(trend_init)
            x = res + trend
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        else:
            x = x.permute(0,2,1)    # x: [Batch, Channel, Input length]
            x = self.model(x)
            x = x.permute(0,2,1)    # x: [Batch, Input length, Channel]
        return x            
        
        
class Model(nn.Module):
    def __init__(self, configs,num_timesteps=288, num_categories=165, hidden_size=128, embedding_dim=64, num_heads=2, num_layers=2):
        super(Model, self).__init__()
        num_stocks=configs.two_tower_col
        self.target_window = configs.pred_len
        self.tower1 = AutocorrelatedTower(input_size=num_timesteps, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers, 
                                     output_size=num_stocks, embedding_dim=embedding_dim, configs=configs)
        # self.tower2 = EmbeddingMLPTower(input_size=num_categories, embedding_dim=embedding_dim, hidden_size=hidden_size, 
        #                                  output_size=num_stocks, num_timesteps=num_timesteps)
        self.tower2 = featureTower(input_size=num_timesteps, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers, 
                                     output_size=num_stocks, embedding_dim=embedding_dim, configs=configs)
        
        batch_size=configs.batch_size
        pred_len=configs.pred_len
        num_feature=configs.two_tower_col
        con_dropout=configs.con_dropout
        # connector_input_size1 = (batch_size, pred_len, num_feature)
        # connector_input_size2 = (batch_size, pred_len, num_feature)
        # connector_output_size = (batch_size, pred_len, num_feature)
        
        connector_input_size1 = num_feature
        connector_input_size2 = num_feature
        connector_output_size = num_feature
        self.connector = ConnectorLayer(input_size1=connector_input_size1, input_size2=connector_input_size2, output_size=connector_output_size,con_dropout=con_dropout)
        # self.linear_mapping = LinearMappingLayer(input_size=hidden_size * 2, output_size=num_stocks)
        self.output_layer = OutputLayer(input_size=num_stocks, output_size=num_stocks)

    # def forward(self, x1, x2):
    #     x1 = x1.long()
    #     x2 = x2.long()
    #     tower1_output = self.tower1(x1)
    #     tower2_output = self.tower2(x2)
    #     fused_output = self.connector(tower1_output, tower2_output)
    #     mapped_output = self.linear_mapping(fused_output)
    #     final_output = self.output_layer(mapped_output)
    #     return final_output
    def forward(self, x):
        # x = x.long()
        # print('x',x.shape)
        tower1_output = self.tower1(x)
        tower2_output = self.tower2(x)
        # print('tower1_output',tower1_output.shape)
        # print('tower2_output',tower2_output.shape)
        fused_output = self.connector(tower1_output, tower2_output)
        # mapped_output = self.linear_mapping(fused_output)
        # print('fused_output',fused_output.shape)
        # final_output = self.output_layer(mapped_output)
        final_output = self.output_layer(fused_output)
        # final_output = self.output_layer(fused_output)
        # print('final_output',final_output.shape)
        return final_output

# class EmbeddingMLPTower(nn.Module):
#     def __init__(self, input_size, embedding_dim, hidden_size, output_size, num_timesteps):
#         super(EmbeddingMLPTower, self).__init__()
#         self.embedding = nn.Linear(input_size, embedding_dim)
#         self.flatten = nn.Flatten()
#         self.mlp = nn.Sequential(
#             nn.Linear(embedding_dim * num_timesteps, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, output_size)
#         )
    
#     def forward(self, x):
#         x = x.float()
#         x = self.embedding(x)
#         x = self.flatten(x)
#         return self.mlp(x)
        # return x
# class EmbeddingMLPTower(nn.Module):
#     def __init__(self, input_size, embedding_dim, hidden_size,output_size, num_timesteps):
#         super(EmbeddingMLPTower, self).__init__()
#         self.embedding = nn.Linear(input_size, embedding_dim)
#         self.flatten = nn.Flatten()
#         self.mlp = nn.Sequential(
#             nn.Linear(embedding_dim * num_timesteps, input_size),  # 输出大小与输入大小相同
#             nn.ReLU(),
#             nn.Linear(input_size, input_size)  # 输出大小与输入大小相同
#         )
    
#     def forward(self, x):
#         x = x.float()
#         x = self.embedding(x)
#         x = self.flatten(x)
#         return self.mlp(x)
# 定义DNN模型
class DNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# class DNN(nn.Module):
#     def __init__(self, input_size, output_size, hidden_size=256, num_layers=2):
#         super(DNN, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
        
#         # 定义输入层
#         self.input_layer = nn.Linear(input_size, hidden_size)
        
#         # 定义隐藏层
#         self.hidden_layers = nn.ModuleList()
#         for _ in range(num_layers - 1):
#             self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        
#         # 定义输出层
#         self.output_layer = nn.Linear(hidden_size, output_size)

#     def forward(self, x, pred_len):
#         # x 的形状: [batch_size, seq_len, input_size]
        
#         # 输入层
#         x = torch.relu(self.input_layer(x))
        
#         # 隐藏层
#         for hidden_layer in self.hidden_layers:
#             x = torch.relu(hidden_layer(x))
        
#         # 动态调整输出层的形状
#         batch_size = x.size(0)
#         seq_len = x.size(1)
#         output_size = x.size(2)
        
#         # 重复最后一个隐藏层的输出，使其与预测长度相匹配
#         x = x.unsqueeze(1).repeat(1, pred_len, 1)
        
#         # 输出层
#         x = self.output_layer(x)
        
#         return x

# class ConnectorLayer(nn.Module):
#     def __init__(self, input_size1, input_size2, output_size):
#         super(ConnectorLayer, self).__init__()
#         self.fc1 = nn.Linear(input_size1, output_size)
#         self.fc2 = nn.Linear(input_size2, output_size)

#     def forward(self, x1, x2):
#         return torch.cat([self.fc1(x1), self.fc2(x2)], dim=1)
    
# class ConnectorLayer(nn.Module):
#     def __init__(self, input_size1, input_size2, output_size):
#         super(ConnectorLayer, self).__init__()
#         # 增加x1的影响，可以将其输入经过更多的全连接层以增加其影响
#         self.fc1_1 = nn.Linear(input_size1, output_size)
#         self.fc1_2 = nn.Linear(output_size, output_size)  # 可以添加额外的全连接层以增加非线性
#         self.fc2 = nn.Linear(input_size2, output_size)

#     def forward(self, x1, x2):
#         # 对x1进行多层全连接处理
#         x1 = self.fc1_1(x1)
#         x1 = torch.relu(x1)  # 添加非线性激活函数
#         x1 = self.fc1_2(x1)  # 再次进行全连接处理
#         # 对x2进行全连接处理
#         x2 = self.fc2(x2)
#         # 将两个处理后的结果进行拼接
#         return torch.cat([x1, x2], dim=1)

#######方案１    
# class ConnectorLayer(nn.Module):
#     def __init__(self, input_size1, input_size2, output_size):
#         super(ConnectorLayer, self).__init__()
#         # 增加x1的影响，可以将其输入经过更多的全连接层以增加其影响
#         self.weight1 = nn.Parameter(torch.Tensor(input_size1))  # 第一个输入张量的权重
#         self.weight2 = nn.Parameter(torch.Tensor(input_size2))  # 第二个输入张量的权重
#         # 初始化权重参数
#         nn.init.uniform_(self.weight1, -3, 3)  # 使用均匀分布初始化权重参数
#         nn.init.uniform_(self.weight2, -1, 1)  # 使用均匀分布初始化权重参数

#     def forward(self, x1,x2):
#         output = self.weight1 * x1 + self.weight2 * x2
#         return output

############方案２
# class ConnectorLayer(nn.Module):
#     def __init__(self, input_size1, input_size2, output_size):
#         super(ConnectorLayer, self).__init__()
#         # 增加x1的影响，可以将其输入经过更多的全连接层以增加其影响
#         hidden_size = min(input_size1, input_size2) // 2  # 设置隐藏层大小为输入大小的一半
#         self.fc1 = nn.Linear(input_size1, hidden_size)  # 第一个输入张量的全连接层
#         self.fc2 = nn.Linear(input_size2, hidden_size)  # 第二个输入张量的全连接层
#         self.fc3 = nn.Linear(hidden_size, output_size)  # 输出层

#         # 初始化全连接层参数
#         nn.init.kaiming_normal_(self.fc1.weight)
#         nn.init.kaiming_normal_(self.fc2.weight)
#         nn.init.kaiming_normal_(self.fc3.weight)

#     def forward(self, x1, x2):
#         # 将输入通过全连接层并使用ReLU激活函数
#         x1 = torch.relu(self.fc1(x1))
#         x2 = torch.relu(self.fc2(x2))
#         # 将两个输入张量的输出相加
#         output = self.fc3(x1 + x2)
#         return output

############方案3
# class ConnectorLayer(nn.Module):
#     def __init__(self, input_size1, input_size2, output_size):
#         super(ConnectorLayer, self).__init__()
#         # 增加x1的影响，可以将其输入经过更多的全连接层以增加其影响
#         hidden_size1 = input_size1 * 2  # 设置第一个隐藏层大小为输入大小的两倍
#         hidden_size2 = input_size2 * 2  # 设置第二个隐藏层大小为输入大小的两倍
#         hidden_size3 = min(hidden_size1, hidden_size2)  # 设置第三个隐藏层大小为两个隐藏层大小的最小值
#         self.fc1_x1 = nn.Linear(input_size1, hidden_size1)  # 第一个输入张量的全连接层
#         self.fc2_x1 = nn.Linear(hidden_size1, hidden_size2)  # 第一个隐藏层
#         self.fc3_x1 = nn.Linear(hidden_size2, hidden_size3)  # 第二个隐藏层
#         self.fc1_x2 = nn.Linear(input_size2, hidden_size1)  # 第二个输入张量的全连接层
#         self.fc2_x2 = nn.Linear(hidden_size1, hidden_size2)  # 第三个隐藏层
#         self.fc3_x2 = nn.Linear(hidden_size2, hidden_size3)  # 第四个隐藏层
#         self.fc4 = nn.Linear(hidden_size3, output_size)  # 输出层

#         # 初始化全连接层参数
#         nn.init.kaiming_normal_(self.fc1_x1.weight)
#         nn.init.kaiming_normal_(self.fc2_x1.weight)
#         nn.init.kaiming_normal_(self.fc3_x1.weight)
#         nn.init.kaiming_normal_(self.fc1_x2.weight)
#         nn.init.kaiming_normal_(self.fc2_x2.weight)
#         nn.init.kaiming_normal_(self.fc3_x2.weight)
#         nn.init.kaiming_normal_(self.fc4.weight)

#     def forward(self, x1, x2):
#         # 将第一个输入通过全连接层并使用ReLU激活函数
#         x1 = torch.relu(self.fc1_x1(x1))
#         x1 = torch.relu(self.fc2_x1(x1))
#         x1 = torch.relu(self.fc3_x1(x1))

#         # 将第二个输入通过全连接层并使用ReLU激活函数
#         x2 = torch.relu(self.fc1_x2(x2))
#         x2 = torch.relu(self.fc2_x2(x2))
#         x2 = torch.relu(self.fc3_x2(x2))

#         # 将两个输入张量的输出相加
#         x = x1 + x2

#         # 通过最后一个隐藏层得到最终输出
#         output = self.fc4(x)
#         return output    
############方案4
# class ConnectorLayer(nn.Module):
#     def __init__(self, input_size1, input_size2, output_size):
#         super(ConnectorLayer, self).__init__()
#         # 增加x1的影响，可以将其输入经过更多的全连接层以增加其影响
#         hidden_size1 = input_size1 * 2  # 设置第一个隐藏层大小为输入大小的两倍
#         hidden_size2 = input_size2 * 2  # 设置第二个隐藏层大小为输入大小的两倍
#         hidden_size3 = min(hidden_size1, hidden_size2) // 2  # 设置第三个隐藏层大小为两个隐藏层大小的一半
#         self.fc1_x1 = nn.Linear(input_size1, hidden_size1)  # 第一个输入张量的全连接层
#         self.fc2_x1 = nn.Linear(hidden_size1, hidden_size2)  # 第一个隐藏层
#         self.fc3_x1 = nn.Linear(hidden_size2, hidden_size3)  # 第二个隐藏层
#         self.fc1_x2 = nn.Linear(input_size2, hidden_size1)  # 第二个输入张量的全连接层
#         self.fc2_x2 = nn.Linear(hidden_size1, hidden_size2)  # 第三个隐藏层
#         self.fc3_x2 = nn.Linear(hidden_size2, hidden_size3)  # 第四个隐藏层
#         self.fc4 = nn.Linear(hidden_size3, output_size)  # 输出层
#         self.leaky_relu = nn.LeakyReLU()  # 使用 LeakyReLU 激活函数

#         # 初始化全连接层参数
#         nn.init.kaiming_normal_(self.fc1_x1.weight)
#         nn.init.kaiming_normal_(self.fc2_x1.weight)
#         nn.init.kaiming_normal_(self.fc3_x1.weight)
#         nn.init.kaiming_normal_(self.fc1_x2.weight)
#         nn.init.kaiming_normal_(self.fc2_x2.weight)
#         nn.init.kaiming_normal_(self.fc3_x2.weight)
#         nn.init.kaiming_normal_(self.fc4.weight)

#     def forward(self, x1, x2):
#         # 将第一个输入通过全连接层并使用 LeakyReLU 激活函数
#         x1 = self.leaky_relu(self.fc1_x1(x1))
#         x1 = self.leaky_relu(self.fc2_x1(x1))
#         x1 = self.leaky_relu(self.fc3_x1(x1))

#         # 将第二个输入通过全连接层并使用 LeakyReLU 激活函数
#         x2 = self.leaky_relu(self.fc1_x2(x2))
#         x2 = self.leaky_relu(self.fc2_x2(x2))
#         x2 = self.leaky_relu(self.fc3_x2(x2))

#         # 将两个输入张量的输出相加
#         x = x1 + x2

#         # 通过最后一个隐藏层得到最终输出
#         output = self.fc4(x)
#         return output
############方案5    
# class ConnectorLayer(nn.Module):
#     def __init__(self, input_size1, input_size2, output_size):
#         super(ConnectorLayer, self).__init__()
#         # 增加x1的影响，减少x2的影响
#         hidden_size1 = input_size1 * 2  # 设置第一个隐藏层大小为输入大小的两倍
#         hidden_size2 = input_size2 // 2  # 设置第二个隐藏层大小为输入大小的一半
#         hidden_size3 = hidden_size1 // 4  # 设置第三个隐藏层大小为第一个隐藏层大小的四分之一
#         self.fc1_x1 = nn.Linear(input_size1, hidden_size1)  # 第一个输入张量的全连接层
#         self.fc2_x1 = nn.Linear(hidden_size1, hidden_size2)  # 第一个隐藏层
#         self.fc3_x1 = nn.Linear(hidden_size2, hidden_size3)  # 第二个隐藏层
#         self.fc4 = nn.Linear(hidden_size3, output_size)  # 输出层
#         self.fc1_x2 = nn.Linear(input_size2, hidden_size1)  # 第二个输入张量的全连接层
#         self.fc2_x2 = nn.Linear(hidden_size1, hidden_size2)  # 第三个隐藏层
#         self.fc3_x2 = nn.Linear(hidden_size2, hidden_size3)  # 第四个隐藏层
#         self.leaky_relu = nn.LeakyReLU()  # 使用 LeakyReLU 激活函数

#         # 初始化全连接层参数
#         nn.init.kaiming_normal_(self.fc1_x1.weight)
#         nn.init.kaiming_normal_(self.fc2_x1.weight)
#         nn.init.kaiming_normal_(self.fc3_x1.weight)
#         nn.init.kaiming_normal_(self.fc4.weight)
#         nn.init.kaiming_normal_(self.fc1_x2.weight)
#         nn.init.kaiming_normal_(self.fc2_x2.weight)
#         nn.init.kaiming_normal_(self.fc3_x2.weight)

#     def forward(self, x1, x2):
#         # 将第一个输入通过全连接层并使用 LeakyReLU 激活函数
#         x1 = self.leaky_relu(self.fc1_x1(x1))
#         x1 = self.leaky_relu(self.fc2_x1(x1))
#         x1 = self.leaky_relu(self.fc3_x1(x1))

#         # 将第二个输入通过全连接层并使用 LeakyReLU 激活函数
#         x2 = self.leaky_relu(self.fc1_x2(x2))
#         x2 = self.leaky_relu(self.fc2_x2(x2))
#         x2 = self.leaky_relu(self.fc3_x2(x2))

#         # 将两个输入张量的输出相加
#         x = x1 + x2

#         # 通过最后一个隐藏层得到最终输出
#         output = self.fc4(x)
#         return output
    

############方案6


class ConnectorLayer(nn.Module):
    def __init__(self, input_size1, input_size2, output_size,con_dropout):
        super(ConnectorLayer, self).__init__()
        # 增加x1的影响，减少x2的影响
        hidden_size1_x1 = input_size1 * 4  # 设置第一个输入张量的第一个隐藏层大小为输入大小的四倍
        hidden_size2_x1 = hidden_size1_x1 // 4  # 设置第一个输入张量的第二个隐藏层大小为第一个隐藏层大小的一半
        hidden_size3_x1 = hidden_size1_x1 // 2  # 设置第一个输入张量的第三个隐藏层大小为第一个隐藏层大小的四分之一
        hidden_size1_x2 = input_size2 * 2  # 设置第二个输入张量的第一个隐藏层大小为输入大小的两倍
        hidden_size2_x2 = hidden_size1_x2 // 2  # 设置第二个输入张量的第二个隐藏层大小为第一个隐藏层大小的一半
        hidden_size3_x2 = hidden_size1_x2 // 2  # 设置第二个输入张量的第三个隐藏层大小为第一个隐藏层大小的四分之一
        self.dropout=nn.Dropout(con_dropout)
        # 第一个输入张量的全连接层
        self.fc1_x1 = nn.Linear(input_size1, hidden_size1_x1)
        # 第一个输入张量的隐藏层
        self.fc2_x1 = nn.Linear(hidden_size1_x1, hidden_size2_x1)
        self.fc3_x1 = nn.Linear(hidden_size2_x1, hidden_size3_x1)
        # 第二个输入张量的全连接层
        self.fc1_x2 = nn.Linear(input_size2, hidden_size1_x2)
        # 第二个输入张量的隐藏层
        self.fc2_x2 = nn.Linear(hidden_size1_x2, hidden_size2_x2)
        self.fc3_x2 = nn.Linear(hidden_size2_x2, hidden_size3_x2)
        
        # 共享的输出层
        self.fc4 = nn.Linear(hidden_size3_x1 + hidden_size3_x2, output_size)
        
        # 使用 LeakyReLU 激活函数
        self.leaky_relu = nn.LeakyReLU()
        
        # 初始化全连接层参数
        nn.init.kaiming_normal_(self.fc1_x1.weight)
        nn.init.kaiming_normal_(self.fc2_x1.weight)
        nn.init.kaiming_normal_(self.fc3_x1.weight)
        nn.init.kaiming_normal_(self.fc1_x2.weight)
        nn.init.kaiming_normal_(self.fc2_x2.weight)
        nn.init.kaiming_normal_(self.fc3_x2.weight)
        nn.init.kaiming_normal_(self.fc4.weight)

    def forward(self, x1, x2):
        # 第一个输入张量的处理
        x1 = self.leaky_relu(self.fc1_x1(x1))
        x = self.dropout(x1)
        x1 = self.leaky_relu(self.fc2_x1(x1))
        x = self.dropout(x1)
        x1 = self.leaky_relu(self.fc3_x1(x1))
        
        # 第二个输入张量的处理
        x2 = self.leaky_relu(self.fc1_x2(x2))
        x = self.dropout(x2)
        x2 = self.leaky_relu(self.fc2_x2(x2))
        x = self.dropout(x2)
        x2 = self.leaky_relu(self.fc3_x2(x2))
        
        # 合并处理后的 x1 和 x2
        x = torch.cat((x1, x2), dim=2)
        x = self.dropout(x)
        # 通过输出层得到最终输出
        output = self.fc4(x)
        
        return output
############方案7
# class ConnectorLayer(nn.Module):
#     def __init__(self, input_size1, input_size2, output_size):
#         super(ConnectorLayer, self).__init__()
#         # Reduce model complexity by using smaller hidden layer sizes
#         hidden_size1_x1 = input_size1 * 2  
#         hidden_size1_x2 = input_size2

#         # Layers for input x1
#         self.fc1_x1 = nn.Linear(input_size1, hidden_size1_x1)
#         self.fc2_x1 = nn.Linear(hidden_size1_x1, output_size)

#         # Layers for input x2
#         self.fc1_x2 = nn.Linear(input_size2, hidden_size1_x2)
#         self.fc2_x2 = nn.Linear(hidden_size1_x2, output_size)

#         # Dropout layer
#         self.dropout = nn.Dropout(0.5)

#         # Initialize weights
#         self.init_weights()

#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Linear):
#                 nn.init.kaiming_normal_(m.weight)
#                 nn.init.zeros_(m.bias)

#     def forward(self, x1, x2):
#         # Processing for input x1
#         x1 = torch.relu(self.fc1_x1(x1))
#         x1 = self.dropout(x1)
#         x1 = torch.relu(self.fc2_x1(x1))
#         x1 = self.dropout(x1)

#         # Processing for input x2
#         x2 = torch.relu(self.fc1_x2(x2))
#         x2 = self.dropout(x2)
#         x2 = torch.relu(self.fc2_x2(x2))
#         x2 = self.dropout(x2)

#         # Concatenate processed x1 and x2
#         x = torch.cat((x1, x2), dim=1)

#         return x
# class ConnectorLayer(nn.Module):
#     def __init__(self, input_size1, input_size2, output_size):
#         super(ConnectorLayer, self).__init__()
#         # 对 x1 和 x2 进行全连接处理
#         self.fc1_1 = nn.Linear(input_size1, output_size)
#         self.fc1_2 = nn.Linear(input_size2, output_size)  # 与 x1 的处理保持一致
#         self.fc_combine = nn.Linear(output_size * 2, output_size)  # 将处理后的 x1 和 x2 合并

#     def forward(self, x1, x2):
#         # 对 x1 进行全连接处理
#         x1 = self.fc1_1(x1)
#         # 对 x2 进行全连接处理
#         x2 = self.fc1_2(x2)
#         # 将处理后的 x1 和 x2 合并
#         combined = torch.cat([x1, x2], dim=1)
#         # 将合并后的结果进行全连接处理
#         output = self.fc_combine(combined)
#         return output    
    
############方案8  tuning0406  test 50times best dropout  0.5 0.3 0.2 0.2
# class ConnectorLayer(nn.Module):
#     def __init__(self, input_size1, input_size2, output_size,con_dropout):
#         super(ConnectorLayer, self).__init__()
#         # 增加x1的影响，减少x2的影响
#         hidden_size1_x1 = input_size1 *2  # 设置第一个输入张量的第一个隐藏层大小为输入大小的四倍
#         hidden_size1_x2 = input_size2   # 设置第二个输入张量的第一个隐藏层大小为输入大小的两倍

#         self.dropout = nn.Dropout(con_dropout)
#         # 第一个输入张量的全连接层
#         self.fc1_x1 = nn.Linear(input_size1, hidden_size1_x1)
#         # 第一个输入张量的隐藏层（去掉了一个隐藏层）
#         self.fc2_x1 = nn.Linear(hidden_size1_x1, output_size)
#         # 第二个输入张量的全连接层
#         self.fc1_x2 = nn.Linear(input_size2, hidden_size1_x2)
#         # 第二个输入张量的隐藏层（去掉了一个隐藏层）
#         self.fc2_x2 = nn.Linear(hidden_size1_x2, output_size)
        
#         # 共享的输出层
#         self.fc3 = nn.Linear(output_size * 2, output_size)
        
#         # 使用 LeakyReLU 激活函数
#         self.leaky_relu = nn.LeakyReLU()
        
#         # 初始化全连接层参数
#         nn.init.kaiming_normal_(self.fc1_x1.weight)
#         nn.init.kaiming_normal_(self.fc2_x1.weight)
#         nn.init.kaiming_normal_(self.fc1_x2.weight)
#         nn.init.kaiming_normal_(self.fc2_x2.weight)
#         nn.init.kaiming_normal_(self.fc3.weight)

#     def forward(self, x1, x2):
#         # 第一个输入张量的处理
#         x1 = self.leaky_relu(self.fc1_x1(x1))
#         x1 = self.dropout(x1)
#         x1 = self.leaky_relu(self.fc2_x1(x1))
        
#         # 第二个输入张量的处理
#         x2 = self.leaky_relu(self.fc1_x2(x2))
#         x2 = self.dropout(x2)
#         x2 = self.leaky_relu(self.fc2_x2(x2))
        
#         # 合并处理后的 x1 和 x2
#         x = torch.cat((x1, x2), dim=2)
#         x = self.dropout(x)
        
#         # 通过输出层得到最终输出
#         output = self.fc3(x)
        
#         return output    
# class LinearMappingLayer(nn.Module):
#     def __init__(self, input_size, output_size):
#         super(LinearMappingLayer, self).__init__()
#         self.flatten = nn.Flatten()
#         self.fc = nn.Linear(input_size, output_size)

#     def forward(self, x):
#         x = self.flatten(x)
#         return self.fc(x)
############方案9   tuning0405  test 50times best dropout 0.5 0.3 0.1 0.2

# class ConnectorLayer(nn.Module):
#     def __init__(self, input_size1, input_size2, output_size,con_dropout):
#         super(ConnectorLayer, self).__init__()
#         # 增加x1的影响，可以将其输入经过更多的全连接层以增加其影响
#         hidden_size = min(input_size1, input_size2) * 2  # 设置隐藏层大小为输入大小的一半
#         self.fc1 = nn.Linear(input_size1, hidden_size)  # 第一个输入张量的全连接层
#         self.fc2 = nn.Linear(input_size2, hidden_size)  # 第二个输入张量的全连接层
#         self.fc3 = nn.Linear(hidden_size, output_size)  # 输出层

#         # Dropout layer
#         self.dropout = nn.Dropout(con_dropout)

#         # 初始化全连接层参数
#         nn.init.kaiming_normal_(self.fc1.weight)
#         nn.init.kaiming_normal_(self.fc2.weight)
#         nn.init.kaiming_normal_(self.fc3.weight)

#     def forward(self, x1, x2):
#         # 将输入通过全连接层并使用ReLU激活函数
#         x1 = torch.relu(self.fc1(x1))
#         x1 = self.dropout(x1)
#         x2 = torch.relu(self.fc2(x2))
#         x2 = self.dropout(x2)
#         # 将两个输入张量的输出相加
#         output = self.fc3(x1 + x2)
#         return output


class OutputLayer(nn.Module):
    def __init__(self, input_size,  output_size):
        super(OutputLayer, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)  # 第一个全连接层
        self.relu = nn.ReLU()  # 非线性激活函数
        self.fc2 = nn.Linear(128, 128)  # 第二个全连接层
        self.fc3 = nn.Linear(128, output_size)  # 输出层

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for prices, categories in train_loader:
            optimizer.zero_grad()
            prices = torch.clip(prices, 0, 9)  # 将负数索引修正为0
            categories = torch.clip(categories, 0, 9)  # 将负数索引修正为0
            output = model(prices, categories)
            prices = prices.mean(dim=1)
            loss = criterion(output, prices)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * prices.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        # 在验证集上评估模型
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for prices, categories in val_loader:
                prices = torch.clip(prices, 0, 9)  # 将负数索引修正为0
                categories = torch.clip(categories, 0, 9)  # 将负数索引修正为0
                output = model(prices, categories)
                prices = prices.mean(dim=1)
                val_loss += criterion(output, prices).item() * prices.size(0)
        
        val_loss /= len(val_loader.dataset)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss}, Val Loss: {val_loss}')
