import torch
import torch.nn as nn
import logging
from typing import List
from ..layers import create_linearblock
from ...utils import get_missing_parameters_message, get_unexpected_parameters_message
from ..build import MODELS, build_model_from_cfg
from ...loss import build_criterion_from_cfg
from ...utils import load_checkpoint
from ...utils.utils_3shape import rotation_6d_to_matrix

@MODELS.register_module()
class BaseReg(nn.Module):
    def __init__(self,
                 encoder_args=None,
                 cls_args=None,
                 criterion_args=None,
                 **kwargs):
        super().__init__()
        self.encoder = build_model_from_cfg(encoder_args)

        if cls_args is not None:
            in_channels = self.encoder.out_channels if hasattr(self.encoder, 'out_channels') else cls_args.get('in_channels', None)
            cls_args.in_channels = in_channels
            self.prediction = build_model_from_cfg(cls_args)
        else:
            self.prediction = nn.Identity()
        self.criterion = build_criterion_from_cfg(criterion_args) if criterion_args is not None else None
        g = 2

    def forward(self, data):
        global_feat = self.encoder.forward_cls_feat(data)
        return self.prediction(global_feat)

    def get_loss(self, pred, gt, normals):
        return self.criterion(pred, gt, normals)

    def get_val_loss(self, pred, gt, normals):
        return self.criterion(pred, gt, normals, True)

@MODELS.register_module()
class RegHead(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mlps: List[int]=[256],
                 norm_args: dict=None,
                 act_args: dict={'act': 'relu'},
                 dropout: float=0.5,
                 global_feat: str=None,
                 point_dim: int=2,
                 **kwargs
                 ):
        """A general classification head. supports global pooling and [CLS] token
        Args:
            in_channels (int): input channels size
            mlps (List[int], optional): channel sizes for hidden layers. Defaults to [256].
            norm_args (dict, optional): dict of configuration for normalization. Defaults to None.
            act_args (_type_, optional): dict of configuration for activation. Defaults to {'act': 'relu'}.
            dropout (float, optional): use dropout when larger than 0. Defaults to 0.5.
            cls_feat (str, optional): preprocessing input features to obtain global feature.
                                      $\eg$ cls_feat='max,avg' means use the concatenateion of maxpooled and avgpooled features.
                                      Defaults to None, which means the input feautre is the global feature
        Returns:
            logits: (B, num_classes, N)
        """
        OUTPUTS_CNT = 6 + 3 # 6 outputs to represent rotation, and 3 for translation
        super().__init__()
        if kwargs:
            logging.warning(f"kwargs: {kwargs} are not used in {__class__.__name__}")
        self.global_feat = global_feat.split(',') if global_feat is not None else None
        self.point_dim = point_dim
        in_channels = len(self.global_feat) * in_channels if global_feat is not None else in_channels
        if mlps is not None:
            mlps = [in_channels] + mlps + [OUTPUTS_CNT]
        else:
            mlps = [in_channels, OUTPUTS_CNT]

        heads = []
        for i in range(len(mlps) - 2):
            heads.append(create_linearblock(mlps[i], mlps[i + 1],
                                            norm_args=norm_args,
                                            act_args=act_args))
            if dropout:
                heads.append(nn.Dropout(dropout))
        heads.append(create_linearblock(mlps[-2], mlps[-1], act_args=None))
        self.head = nn.Sequential(*heads)


    def forward(self, end_points):
        if self.global_feat is not None:
            global_feats = []
            for preprocess in self.global_feat:
                if 'max' in preprocess:
                    global_feats.append(torch.max(end_points, dim=self.point_dim, keepdim=False)[0])
                elif preprocess in ['avg', 'mean']:
                    global_feats.append(torch.mean(end_points, dim=self.point_dim, keepdim=False))
            end_points = torch.cat(global_feats, dim=1)
        head_out = self.head(end_points)
        rotation_6d = rotation_6d_to_matrix(head_out[:, :6])
        translation = head_out[:, 6:]
        batch_size = head_out.shape[0]
        I = torch.eye(4, device=head_out.device, dtype=head_out.dtype)
        homographies = I.repeat(batch_size, 1, 1)
        homographies[:, :3, :3] = rotation_6d
        homographies[:, :3, 3] = translation
        return homographies
