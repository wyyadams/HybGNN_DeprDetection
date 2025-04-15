import torch
import torch.nn.functional as F
from torch import nn,einsum
import math


class FeatureEncoder(nn.Module):
    def __init__(self):
        super(FeatureEncoder, self).__init__()
        self.Conv1d_block1_1 = nn.Sequential(
            nn.Conv1d(1, 16, 16, 2, 0),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(16, 16),
        )

        self.Conv1d_block1_2 = nn.Sequential(
            nn.Conv1d(16, 32, 8, 1, 4),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 7, 1, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 7, 1, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, 4, 0),
            nn.Flatten()
        )

        self.Conv1d_block2_1 = nn.Sequential(
            nn.Conv1d(1, 32, 64, 8, 0),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(3, 3),
        )

        self.Conv1d_block2_2 = nn.Sequential(
            nn.Conv1d(32, 32, 7, 1, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 7, 1, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 32, 7, 1, 3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4, 4, 0),
            nn.Flatten()
        )

    def forward(self, x):
        bs, chan, sample = x.shape
        x = x.reshape(bs * chan, 1, sample)
        x1 = self.Conv1d_block1_1(x)
        x1 = self.Conv1d_block1_2(x1)
        x2 = self.Conv1d_block2_1(x)
        x2 = self.Conv1d_block2_2(x2)
        x_out = torch.cat((x1, x2), dim=-1)
        x_out = x_out.reshape(bs, chan, -1)
        return x_out


def normalize_A(A):
    A = F.relu(A)
    N = A.shape[0]
    A = A * (torch.ones(N, N, device=A.device) - torch.eye(N, N, device=A.device))
    A = A + A.T
    A = A + torch.eye(A.shape[0], device=A.device)
    d = torch.sum(A, 1)
    d = 1 / torch.sqrt((d + 1e-8))
    D = torch.diag_embed(d)
    Lnorm = torch.matmul(torch.matmul(D, A), D)
    return Lnorm


def BatchAdjNorm(A):
    A = F.relu(A)
    bs, n, _ = A.shape
    identity = torch.eye(n, n, device=A.device)
    identity_matrix = identity.repeat(bs, 1, 1)
    A = A * (torch.ones(bs, n, n, device=A.device) - identity_matrix)
    A = A + A.transpose(1, 2)
    A = A + identity_matrix
    d = torch.sum(A, 2)
    d = 1 / torch.sqrt((d + 1e-8))
    D = torch.diag_embed(d)
    Lnorm = torch.matmul(torch.matmul(D, A), D)
    return Lnorm


class AdjGenerator(nn.Module):
    def __init__(self, in_dim, dim_hid=32, Tem=1):
        super().__init__()
        self.inner_dim = dim_hid
        self.scale = Tem ** -1
        self.attend = nn.Softmax(dim=-1)
        self.to_q = nn.Linear(in_dim, self.inner_dim, bias=False)
        self.to_k = nn.Linear(in_dim, self.inner_dim, bias=False)
        nn.init.xavier_normal_(self.to_q.weight)
        nn.init.xavier_normal_(self.to_k.weight)

    def forward(self, x):
        b, n, _ = x.shape
        q = self.to_q(x)
        k = self.to_k(x)
        dots = einsum('b i d, b j d -> b i j', q, k) * self.scale
        attn = self.attend(dots)
        return attn


class GraphConvolution(nn.Module):

    def __init__(self, in_channels, out_channels, device, bias=False, init='xavier'):

        super(GraphConvolution, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(device))
        self.init = init
        self._para_init()
        self.bias = None
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_channels).to(device))
            nn.init.zeros_(self.bias)

    def _para_init(self):
        if self.init == 'xavier':
            nn.init.xavier_normal_(self.weight)
        else:
            nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x, adj):
        out = torch.matmul(adj, x)
        out = torch.matmul(out, self.weight)
        if self.bias is not None:
            return out + self.bias
        else:
            return out


class GCN_Concat(nn.Module):
    def __init__(self, device, in_channels, out_channels, K):
        super(GCN_Concat, self).__init__()
        self.K = K
        self.gnn = nn.ModuleList()
        for i in range(self.K):
            self.gnn.append(GraphConvolution(in_channels, out_channels, device))

    def forward(self, x, L):
        adj = self.generate_adj(L, self.K)
        out = None
        for i in range(len(self.gnn)):
            if i == 0:
                out = F.leaky_relu(self.gnn[i](x, adj[i]))
            else:
                out += F.leaky_relu(self.gnn[i](x, adj[i]))
        return out

    def generate_adj(self, L, K):
        support = []
        L_iter = L
        for i in range(K):
            if i == 0:
                support.append(torch.eye(L.shape[-1]).cuda())
            else:
                support.append(L_iter)
                L_iter = L_iter * L
        return support


class GCN(nn.Module):
    def __init__(self, device, in_channels, out_channels, K):
        super(GCN, self).__init__()
        self.K = K
        self.gnn = GraphConvolution(in_channels, out_channels, device)

    def forward(self, x, L):
        adj = self.generate_adj(L, self.K)
        out = self.gnn(x, adj[self.K - 1])
        return out

    def generate_adj(self, L, K):
        support = []
        L_iter = L
        for i in range(K):
            if i == 0:
                support.append(torch.eye(L.shape[-1]).cuda())
            else:
                support.append(L_iter)
                L_iter = L_iter * L
        return support


class FGNN(nn.Module):
    def __init__(self, device, in_dim, gnn_out_dim, gnn_k=1):
        super().__init__()
        self.FGCN = GCN(device, in_dim, gnn_out_dim, gnn_k)

    def forward(self, x, Common_A):
        x = self.FGCN(x, Common_A)
        return x


class Assign_filter(nn.Module):
    def __init__(self, chan_num, feat_dim, reg_num):
        super().__init__()
        self.Proj = nn.Linear(chan_num, feat_dim, bias=False)
        self.Propa = nn.Linear(feat_dim * 2, reg_num, bias=False)
        nn.init.xavier_normal_(self.Proj.weight)
        nn.init.xavier_normal_(self.Propa.weight)

    def forward(self, feat_fixed, Common_A):
        bs, _, _ = feat_fixed.shape
        Proj_Common_A = self.Proj(Common_A.repeat(bs, 1, 1))
        embed = torch.concat((Proj_Common_A, feat_fixed), dim=-1)
        return embed


class SE_layer(nn.Module):
    def __init__(self, robust_emb_dim, x_dim, ratio):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(robust_emb_dim, robust_emb_dim // ratio, bias=False),
            nn.ReLU(),
            nn.Linear(robust_emb_dim // ratio, x_dim, bias=False),
            nn.Sigmoid()
        )
        self.x_dim = x_dim

        for m in self.fc:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, robust_emb, x):
        bs, chan, dim = robust_emb.shape
        robust_emb = robust_emb.transpose(1, 2)
        robust_emb_y = self.gap(robust_emb).view(bs, dim)
        x_weight = self.fc(robust_emb_y).view(bs, self.x_dim, 1).expand_as(x.transpose(1, 2))
        x_weight = x_weight.transpose(1, 2)
        return x * x_weight


class SE_assign(nn.Module):
    def __init__(self, robust_emb_dim, x_dim, reg_num, ratio):
        super().__init__()
        self.SE = SE_layer(robust_emb_dim, x_dim, ratio)
        self.fc = nn.Linear(x_dim, reg_num, bias=False)
        nn.init.xavier_normal_(self.fc.weight)

    def forward(self, robust_emb, x, A_indivi):
        x = torch.matmul(A_indivi, x)
        x = self.SE(robust_emb, x)
        S_SE = F.softmax(self.fc(x), dim=-1)
        return S_SE


class FHIE(nn.Module):
    def __init__(self, device, in_dim, gnn_out_dim, reg_num, chan_num, lambda_g, miu_g, SE_r, Reg_gnn_K=1):
        super().__init__()
        self.SE = SE_assign(robust_emb_dim=in_dim * 2, x_dim=in_dim, ratio=SE_r, reg_num=reg_num)
        self.S_filter = Assign_filter(chan_num, gnn_out_dim, reg_num)
        self.reg_GCN = GCN_Concat(device, in_dim, gnn_out_dim, Reg_gnn_K)
        self.lambda_g = lambda_g
        self.miu_g = miu_g

    def forward(self, x, Indivi_A, Common_A, Common_Feat):
        robust_emb = self.S_filter(Common_Feat, Common_A)
        S_SE = self.SE(robust_emb, x, Indivi_A)
        Indivi_feat = torch.matmul(S_SE.transpose(1, 2), x)
        Indivi_reg_graph = torch.matmul(torch.matmul(S_SE.transpose(1, 2), Indivi_A), S_SE)
        reg_emb = self.reg_GCN(Indivi_feat, Indivi_reg_graph)
        HI_emb = torch.matmul(S_SE, reg_emb)
        global_loss = self.loss_compute(Indivi_A, S_SE, self.lambda_g, self.miu_g)
        return HI_emb, global_loss, Indivi_reg_graph

    def loss_compute(self, Indivi_A, S, lambda1, miu):
        S_St = torch.matmul(S, S.transpose(-1, -2))
        diff = Indivi_A - S_St
        Loss_lp = torch.norm(diff, p='fro')

        R_log_R = S * torch.log(S + 1e-10)
        entropy_loss = -R_log_R.sum(dim=(1, 2))  # 对 n 和 n_r 维度求和
        Loss_e = miu * entropy_loss
        loss = lambda1 * Loss_lp + miu * Loss_e
        return loss


class HybGNN(nn.Module):
    def __init__(self, device, node_in_dim, out_dim, reg_num, gc_dim_node, chan_num, FGCN_K, IGCN_K, RGCN_K, lambda_g,
                 miu_g, hyper_l1, SE_R):
        super().__init__()
        self.hyper_l1 = hyper_l1
        self.Encoder = FeatureEncoder()
        self.Common_A = nn.Parameter(torch.FloatTensor(chan_num, chan_num))
        nn.init.xavier_normal_(self.Common_A)
        self.Indivi_A_gene = AdjGenerator(node_in_dim, gc_dim_node)

        self.FGCN = GCN_Concat(device, node_in_dim, out_dim, FGCN_K)
        self.IGCN = GCN_Concat(device, node_in_dim, out_dim, IGCN_K)

        self.FHIE = FHIE(device, out_dim, out_dim, reg_num, chan_num, lambda_g, miu_g, SE_R, RGCN_K)

        self.Flat = nn.Flatten()
        self.to_cls = nn.Linear(out_dim * 3 * chan_num, 2)
        nn.init.xavier_normal_(self.to_cls.weight)

    def forward(self, x):
        feat = self.Encoder(x)
        FGCN_adj = normalize_A(self.Common_A)
        FGNN_Embed = self.FGCN(feat, FGCN_adj)

        IGCN_adj = BatchAdjNorm(self.Indivi_A_gene(feat))
        IGCN_Embed = self.IGCN(feat, IGCN_adj)

        HIE_Embed, HIE_loss, reg_Adj = self.FHIE(IGCN_Embed, IGCN_adj, FGCN_adj, FGNN_Embed)

        Out_Embed = torch.concat((FGNN_Embed, IGCN_Embed, HIE_Embed), dim=-1)
        Out_Embed = self.Flat(Out_Embed)

        cls_out = self.to_cls(Out_Embed)
        loss = self.l1_loss_compute(self.Common_A, reg_Adj, self.hyper_l1) + torch.sum(HIE_loss)
        return cls_out, loss

    def l1_loss_compute(self, Node_Graph, local_reg_graph, hyper):
        node_graph_l1 = torch.norm(Node_Graph, p=1)
        reg_graph_l1 = torch.norm(local_reg_graph, p=1)
        loss = hyper * (node_graph_l1 + reg_graph_l1)
        return loss


if __name__ == '__main__':
    data = torch.rand((128, 19, 1000)).cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net = HybGNN(device, 512, 128, 5, 128, 19, 2, 2, 2, 1e-5, 1e-5, 1e-5, 4).cuda()
    out, loss = net(data)
    print(out.shape)

    total_params = sum(p.numel() for p in net.parameters())
    print(f'Total number of parameters: {total_params}')

    for name, param in net.named_parameters():
        print(f'Layer: {name}, Number of parameters: {param.numel()}')
