# group layer: find neighbors for each point
# knn, knn_sparse, ball query

# gather layer, gather features by index
from typing import Tuple
import copy, logging
import torch
import torch.nn as nn
from torch.autograd import Function
#from openpoints.cpp import pointnet2_cuda
import time
class KNN(nn.Module):
    def __init__(self, neighbors, transpose_mode=True):
        super(KNN, self).__init__()
        self.neighbors = neighbors

    @torch.no_grad()
    def forward(self, support, query):
        """
        Args:
            support ([tensor]): [B, N, C]
            query ([tensor]): [B, M, C]
        Returns:
            [int]: neighbor idx. [B, M, K]
        """
        dist = torch.cdist(support, query)
        k_dist = dist.topk(k=self.neighbors, dim=1, largest=False)
        return k_dist.values, k_dist.indices.transpose(1, 2).contiguous().int()

# dilated knn
class DenseDilated(nn.Module):
    """
    Find dilated neighbor from neighbor list
    index: (B, npoint, nsample)
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DenseDilated, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            if torch.rand(1) < self.epsilon and self.training:
                num = self.k * self.dilation
                randnum = torch.randperm(num)[:self.k]
                edge_index = edge_index[:, :, randnum]
            else:
                edge_index = edge_index[:, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, ::self.dilation]
        return edge_index.contiguous()


class DilatedKNN(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """

    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(DilatedKNN, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self._dilated = DenseDilated(k, dilation, stochastic, epsilon)
        self.knn = KNN(k * self.dilation, transpose_mode=True)

    def forward(self, query):
        _, idx = self.knn(query, query)
        return self._dilated(idx)


class GroupingOperation(torch.jit.ScriptModule):

    def __init__(self):
        super(GroupingOperation, self).__init__()

    @torch.jit.script_method
    def forward(self, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param features: (B, C, N) tensor of features to group
        :param idx: (B, npoint, nsample) tensor containing the indicies of features to group with
        :return:
            output: (B, C, npoint, nsample) tensor
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, nfeatures, nsample = idx.size()
        _, C, N = features.size()
        output = torch.empty(B, C, nfeatures, nsample, device=features.device, dtype=torch.float32)

        torch.ops.my_ops.group_points_wrapper_fast(B, C, N, nfeatures, nsample, features, idx, output)

        # Note: TorchScript doesn't support custom backward operations
        # So ctx.for_backwards = (idx, N) is removed

        return output



grouping_operation = GroupingOperation


def torch_grouping_operation(features, idx):
    r"""from torch points kernels
    Parameters
    ----------
    features : torch.Tensor
        (B, C, N) tensor of features to group
    idx : torch.Tensor
        (B, npoint, nsample) tensor containing the indicies of features to group with

    Returns
    -------
    torch.Tensor
        (B, C, npoint, nsample) tensor
    """
    all_idx = idx.reshape(idx.shape[0], -1)
    all_idx = all_idx.unsqueeze(1).repeat(1, features.shape[1], 1)
    grouped_features = features.gather(2, all_idx)
    return grouped_features.reshape(idx.shape[0], features.shape[1], idx.shape[1], idx.shape[2])


class GatherOperation(Function):

    @staticmethod
    def forward(ctx, features: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        :param ctx:
        :param features: (B, C, N)
        :param idx: (B, npoint) index tensor of the features to gather
        :return:
            output: (B, C, npoint)
        """
        assert features.is_contiguous()
        assert idx.is_contiguous()

        B, npoint = idx.size()
        _, C, N = features.size()
        output = torch.cuda.FloatTensor(B, C, npoint, device=features.device)

        torch.ops.my_ops.gather_points_wrapper(B, C, N, npoint, features, idx, output)

        ctx.for_backwards = (idx, C, N)
        return output

    @staticmethod
    def backward(ctx, grad_out):
        idx, C, N = ctx.for_backwards
        B, npoint = idx.size()

        grad_features = torch.zeros([B, C, N], dtype=torch.float, device=grad_out.device, requires_grad=True)
        grad_out_data = grad_out.data.contiguous()
        torch.ops.my_ops.gather_points_grad_wrapper(B, C, N, npoint, grad_out_data, idx, grad_features.data)
        return grad_features, None


gather_operation = GatherOperation.apply

def ball_query_radius(new_xyz, xyz, radius, nsample):
    """
    Arguments:
    - new_xyz: (B, M, 3) PyTorch Tensor
    - xyz: (B, N, 3) PyTorch Tensor
    - radius: float
    - nsample: int

    Returns:
    - idx: (B, M, nsample) PyTorch Tensor
    """

    B, M, _ = new_xyz.size()
    _, N, _ = xyz.size()

    new_xyz = new_xyz.unsqueeze(2)  # (B, M, 1, 3)
    xyz = xyz.unsqueeze(1)  # (B, 1, N, 3)

    dist2 = ((new_xyz - xyz) ** 2).sum(dim=-1)  # (B, M, N)

    radius2 = radius ** 2
    mask_within_radius = dist2 < radius2  # (B, M, N)

    idx = torch.zeros((B, M, nsample), dtype=torch.long, device=new_xyz.device)  # (B, M, nsample)
    cnt = mask_within_radius.sum(dim=-1)  # (B, M)

    arange = torch.arange(0, N, device=new_xyz.device)
    arange = arange.view(1, 1, -1).expand(B, M, -1)  # (B, M, N)

    idx_within_radius = torch.where(mask_within_radius, arange,
                                    N + torch.ones((B, M, N), dtype=torch.long, device=new_xyz.device))  # (B, M, N)

    for i in range(B):
        for j in range(M):
            idx[i, j, :cnt[i, j].item()] = idx_within_radius[i, j, :cnt[i, j].item()].sort()[0][:nsample]

    return idx


class BallQuery(torch.jit.ScriptModule):

    def __init__(self):
        super(BallQuery, self).__init__()

    @torch.jit.script_method
    def forward(self, radius: float, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor) -> torch.Tensor:
        """
        :param radius: float, radius of the balls
        :param nsample: int, maximum number of features in the balls
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: (B, npoint, 3) centers of the ball query
        :return:
            idx: (B, npoint, nsample) tensor with the indices of the features that form the query balls
        """
        assert new_xyz.is_contiguous()
        assert xyz.is_contiguous()

        B, N, _ = xyz.size()
        npoint = new_xyz.size(1)

        # Use torch.empty instead of direct torch.cuda.IntTensor
        idx = torch.empty(B, npoint, nsample, dtype=torch.int32, device=xyz.device).zero_()
        torch.ops.my_ops.BallQueryGpu(B, N, npoint, radius, nsample, new_xyz, xyz, idx)
        return idx

    @staticmethod
    def backward(ctx, a=None):
        return None, None, None, None


ball_query = BallQuery


class QueryAndGroup(nn.Module):
    def __init__(self, radius: float, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 normalize_by_std=False,
                 normalize_by_allstd=False,
                 normalize_by_allstd2=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """[summary]

        Args:
            radius (float): radius of ball
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.radius, self.nsample = radius, nsample
        self.normalize_dp = normalize_dp
        self.normalize_by_std = normalize_by_std
        self.normalize_by_allstd = normalize_by_allstd
        self.normalize_by_allstd2 = normalize_by_allstd2
        assert self.normalize_dp + self.normalize_by_std + self.normalize_by_allstd < 2   # only nomalize by one method
        self.relative_xyz = relative_xyz
        self.return_only_idx = return_only_idx

        self.q = BallQuery()
        self.grouping_operation = GroupingOperation()

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        """
        :param query_xyz: (B, npoint, 3) xyz coordinates of the features
        :param support_xyz: (B, N, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        #self.q = BallQuery()
        idx = self.q(self.radius, self.nsample, support_xyz, query_xyz)

        if self.return_only_idx:
            return idx
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = self.grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        if self.relative_xyz:
            grouped_xyz = grouped_xyz - query_xyz.transpose(1, 2).unsqueeze(-1)  # relative position
            if self.normalize_dp:
                grouped_xyz /= self.radius
        grouped_features = self.grouping_operation(features, idx) if features is not None else None
        return grouped_xyz, grouped_features


class GroupAll(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, new_xyz: torch.Tensor, xyz: torch.Tensor, features: torch.Tensor = None):
        """
        :param xyz: (B, N, 3) xyz coordinates of the features
        :param new_xyz: ignored
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, C + 3, 1, N)
        """
        grouped_xyz = xyz.transpose(1, 2).unsqueeze(2)
        grouped_features = features.unsqueeze(2) if features is not None else None
        return grouped_xyz, grouped_features


class KNNGroup(nn.Module):
    def __init__(self, nsample: int,
                 relative_xyz=True,
                 normalize_dp=False,
                 return_only_idx=False,
                 **kwargs
                 ):
        """[summary]

        Args:
            nsample (int): maximum number of features to gather in the ball
            use_xyz (bool, optional): concate xyz. Defaults to True.
            ret_grouped_xyz (bool, optional): [description]. Defaults to False.
            normalize_dp (bool, optional): [description]. Defaults to False.
        """
        super().__init__()
        self.nsample = nsample
        self.knn = KNN(nsample, transpose_mode=True)
        self.relative_xyz = relative_xyz
        self.normalize_dp = normalize_dp
        self.return_only_idx = return_only_idx

    def forward(self, query_xyz: torch.Tensor, support_xyz: torch.Tensor, features: torch.Tensor = None) -> Tuple[
        torch.Tensor]:
        """
        :param query_xyz: (B, N, 3) xyz coordinates of the features
        :param support_xyz: (B, npoint, 3) centroids
        :param features: (B, C, N) descriptors of the features
        :return:
            new_features: (B, 3 + C, npoint, nsample)
        """
        _, idx = self.knn(support_xyz, query_xyz)
        if self.return_only_idx:
            return idx
        idx = idx.int()
        xyz_trans = support_xyz.transpose(1, 2).contiguous()
        grouped_xyz = grouping_operation(xyz_trans, idx)  # (B, 3, npoint, nsample)
        if self.relative_xyz:
            grouped_xyz -= query_xyz.transpose(1, 2).unsqueeze(-1)  # relative position
        if self.normalize_dp:
            grouped_xyz /= torch.amax(torch.sqrt(torch.sum(grouped_xyz**2, dim=1)), dim=(1, 2)).view(-1, 1, 1, 1)
        if features is not None:
            grouped_features = grouping_operation(features, idx)
            return grouped_xyz, grouped_features
        else:
            return grouped_xyz, None


def get_aggregation_feautres(p, dp, f, fj, feature_type='dp_fj'):
    if feature_type == 'dp_fj':
        fj = torch.cat([dp, fj], 1)
    elif feature_type == 'dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, fj, df], 1)
    elif feature_type == 'pi_dp_fj_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([p.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, df.shape[-1]), dp, fj, df], 1)
    elif feature_type == 'dp_df':
        df = fj - f.unsqueeze(-1)
        fj = torch.cat([dp, df], 1)
    return fj


def create_grouper(group_args):
    group_args_copy = copy.deepcopy(group_args)
    method = group_args_copy.pop('NAME', 'ballquery')
    radius = group_args_copy.pop('radius', 0.1)
    nsample = group_args_copy.pop('nsample', 20)

    logging.info(group_args)
    if nsample is not None:
        if method == 'ballquery':
            grouper = QueryAndGroup(radius, nsample, **group_args_copy)
        elif method == 'knn':
            grouper = KNNGroup(nsample,  **group_args_copy)
    else:
        grouper = GroupAll()
    return grouper

from openpoints.models.layers.subsample import furthest_point_sample, random_sample

def min_graph():
    B, C, N = 2, 3, 40960
    device = 'cuda'
    points = torch.randn([B, N, C], device=device, dtype=torch.float)
    npoints = 10000
    idx = furthest_point_sample(points, npoints).to(torch.int64)
    return idx
   # query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    #return query

if __name__ == "__main__":
    import time

    B, C, N = 2, 3, 40960
    K = 16
    device = 'cuda'
    points = torch.randn([B, N, C], device=device, dtype=torch.float)
    print(points.shape, '\n', points)

    # --------------- debug downsampling
    #from openpoints.models.layers.layer3d import RandomSample, random_sample, furthest_point_sample
    from openpoints.models.layers.subsample import furthest_point_sample, random_sample
    npoints = 10000
    # rs = RandomSample(num_to_sample=npoints)
    # query, _= rs(points)
    idx = random_sample(points, npoints)
    # torch gather is faster then operation gather.
    query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    print(query.shape, '\n', query)

    idx = furthest_point_sample(points, npoints).to(torch.int64)
    query = torch.gather(points, 1, idx.unsqueeze(-1).expand(-1, -1, 3))
    print(query.shape, '\n', query)

    # # --------------- debug KNN
    # knn = KNN(k=K, transpose_mode=True)
    # # knn to get the neighborhood

    # # compare time usage.
    # st = time.time()
    # for _ in range(100):
    #     _, knnidx = knn(points, query) # B G M
    #     idx_base = torch.arange(0, B, device=points.device).view(-1, 1, 1) * N
    #     idx = knnidx + idx_base
    #     idx = idx.view(-1)
    #     neighborhood = points.view(B * N, -1)[idx, :]
    #     neighborhood = neighborhood.view(B, npoints, K, 3).contiguous()
    #     # normalize
    #     neighborhood1 = neighborhood - query.unsqueeze(2)
    # print(time.time() - st)
    # # print(neighborhood1.shape, '\n', neighborhood1)

    # knngroup = KNNGroup(K)
    # # KNN Group is faster then above torch indexing when warpped in class.
    # st = time.time()
    # for _ in range(100):
    #     neighborhood2 = knngroup(query, points)
    # print(time.time() - st)
    # # print(neighborhood2.shape, '\n', neighborhood2)
    # flag = torch.allclose(neighborhood1, neighborhood2.permute(0, 2, 3, 1))
    # print(flag)

    # ------------- debug ball query
    query_group = QueryAndGroup(0.1, K)

    st = time.time()
    for _ in range(100):
        # ball querying is 40 times faster then KNN
        features = query_group(query, points)
    print(time.time() - st)
    print(features.shape)
