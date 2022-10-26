import argparse
import random
import pickle
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.manifold import TSNE
from torch.utils.data.sampler import Sampler
from typing import Sized
from tqdm import tqdm
from torch import linalg as LA
from pytorch_metric_learning.utils.inference import CustomKNN
from pytorch_metric_learning.distances import CosineSimilarity

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_masked_input_and_labels(inp, mask_value=1, mask_p=0.15, mask_random_p=0.1, mask_remain_p=0.1, mask_random_s=1):
    # BERT masking
    inp_mask = (torch.rand(*inp.shape[:2]) < mask_p).to(inp.device)

    # Prepare input
    inp_masked = inp.clone().float()

    # Set input to [MASK] which is the last token for the 90% of tokens
    # This means leaving 10% unchanged
    inp_mask_2mask = (inp_mask & (torch.rand(*inp.shape[:2]) < 1 - mask_remain_p).to(inp.device))
    inp_masked[inp_mask_2mask] = mask_value # mask token is the last in the dict

    # Set 10% to a random token
    inp_mask_2random = inp_mask_2mask & (torch.rand(*inp.shape[:2]) < mask_random_p / (1 - mask_remain_p)).to(inp.device)

    inp_masked[inp_mask_2random] = (2 * mask_random_s * torch.rand(inp_mask_2random.sum().item(), inp.shape[2]) - mask_random_s).to(inp.device)

    # y_labels would be same as encoded_texts i.e input tokens
    gt = inp.clone()
    return inp_masked, gt

def random_rot_mat(bs, uniform_dist):
    rot_mat = torch.zeros(bs, 3, 3)
    random_values = uniform_dist.rsample((bs,))
    rot_mat[:, 0, 0] = torch.cos(random_values)
    rot_mat[:, 0, 1] = -torch.sin(random_values)
    rot_mat[:, 1, 0] = torch.sin(random_values)
    rot_mat[:, 1, 1] = torch.cos(random_values)
    rot_mat[:, 2, 2] = 1
    return rot_mat

def repeat_rot_mat(rot_mat, num):
    batch = rot_mat.shape[0]
    res = torch.zeros([batch, 3*num, 3*num]).to(rot_mat.device)
    for i in range(num):
        res[:, 3*i:3*(i+1), 3*i:3*(i+1)] = rot_mat
    return res

def align_skeleton(data):
    N, C, T, V, M = data.shape
    trans_data = np.zeros_like(data)
    for i in tqdm(range(N)):
        for p in range(M):
            sample = data[i][..., p]
            # if np.all((sample[:,0,:] == 0)):
                # continue
            d = sample[:,0,1:2]
            v1 = sample[:,0,1]-sample[:,0,0]
            if np.linalg.norm(v1) <= 0.0:
                continue
            v1 = v1/np.linalg.norm(v1)
            v2_ = sample[:,0,12]-sample[:,0,16]
            proj_v2_v1 = np.dot(v1.T,v2_)*v1/np.linalg.norm(v1)
            v2 = v2_-np.squeeze(proj_v2_v1)
            v2 = v2/(np.linalg.norm(v2))
            v3 = np.cross(v2,v1)/(np.linalg.norm(np.cross(v2,v1)))
            v1 = np.reshape(v1,(3,1))
            v2 = np.reshape(v2,(3,1))
            v3 = np.reshape(v3,(3,1))

            R = np.hstack([v2,v3,v1])
            for t in range(T):
                trans_sample = (np.linalg.inv(R))@(sample[:,t,:]) # -d
                trans_data[i, :, t, :, p] = trans_sample
    return trans_data

def create_aligned_dataset(file_list=['data/ntu/NTU60_CS.npz', 'data/ntu/NTU60_CV.npz']):
    for file in file_list:
        org_data = np.load(file)
        splits = ['x_train', 'x_test']
        aligned_set = {}
        for split in splits:
            data = org_data[split]
            N, T, _ = data.shape
            data = data.reshape((N, T, 2, 25, 3)).transpose(0, 4, 1, 3, 2)
            aligned_data = align_skeleton(data)
            aligned_data = aligned_data.transpose(0, 2, 4, 3, 1).reshape(N, T, -1)
            aligned_set[split] = aligned_data

        np.savez(file.replace('.npz', '_aligned.npz'),
                 x_train=aligned_set['x_train'],
                 y_train=org_data['y_train'],
                 x_test=aligned_set['x_test'],
                 y_test=org_data['y_test'])



def get_motion(data, data_format=['x'], use_nonzero_mask=False, rot=False, jittering=False, random_dist=None):
    N, C, T, V, M = data.size()
    data = data.permute(0, 4, 2, 3, 1).contiguous().view(N*M, T, V, C)

    # get motion features
    x = data - data[:,:,0:1,:] # localize
    if 'v' in data_format:
        v = x[:,1:,:,:] - x[:,:-1,:,:]
        v = torch.cat([torch.zeros(N*M, 1, V, C).to(v.device), v], dim=1)
    if 'a' in data_format:
        a = v[:,1:,:,:] - v[:,:-1,:,:]
        a = torch.cat([torch.zeros(N*M, 1, V, C).to(a.device), a], dim=1)

    # reshape x,v for PORT
    x = x.view(N*M*T, V, C)
    if 'v' in data_format:
        v = v.view(N*M*T, V, C)
    if 'a' in data_format:
        a = a.view(N*M*T, V, C)

    # apply nonzero mask
    if use_nonzero_mask:
        nonzero_mask = x.view(N*M*T, -1).count_nonzero(dim=-1) !=0
        x = x[nonzero_mask]
        if 'v' in data_format:
            v = v[nonzero_mask]
        if 'a' in data_format:
            a = a[nonzero_mask]

    # optionally rotate
    if rot:
        rot_mat = random_rot_mat(x.shape[0], random_dist).to(x.device)
        x = x.transpose(1, 2) # (NMT, C, V)
        x = torch.bmm(rot_mat, x) # rotate
        x = x.transpose(1, 2) #(NMT, V, C)

        if 'v' in data_format:
            v = v.transpose(1, 2) # (NMT, C, V)
            v = torch.bmm(rot_mat, v) # rotate
            v = v.transpose(1, 2) #(NMT, V, C)

        if 'a' in data_format:
            a = a.transpose(1, 2) # (NMT, C, V)
            a = torch.bmm(rot_mat, a) # rotate
            a = a.transpose(1, 2) #(NMT, V, C)

    if jittering:
        jit = (torch.rand(x.shape[0], 1, x.shape[-1], device=x.device) - 0.5) / 10
        x += jit

    output = {'x':x}
    if 'v' in data_format:
        output['v'] = v
    if 'a' in data_format:
        output['a'] = a

    return output

def get_attn(x, mask= None, similarity='scaled_dot'):
    if similarity == 'scaled_dot':
        sqrt_dim = np.sqrt(x.shape[-1])
        score = torch.bmm(x, x.transpose(1, 2)) / sqrt_dim
    elif similarity == 'euclidean':
        score = torch.cdist(x, x)

    if mask is not None:
        score.masked_fill_(mask.view(score.size()), -float('Inf'))

    attn = F.softmax(score, -1)
    embd = torch.bmm(attn, x)
    return embd, attn

def get_vector_property(x):
    N, C = x.size()
    x1 = x.unsqueeze(0).expand(N, N, C)
    x2 = x.unsqueeze(1).expand(N, N, C)
    x1 = x1.reshape(N*N, C)
    x2 = x2.reshape(N*N, C)
    cos_sim = F.cosine_similarity(x1, x2, dim=1, eps=1e-6).view(N, N)
    cos_sim = torch.triu(cos_sim, diagonal=1).sum() * 2 / (N*(N-1))
    pdist = (LA.norm(x1-x2, ord=2, dim=1)).view(N, N)
    pdist = torch.triu(pdist, diagonal=1).sum() * 2 / (N*(N-1))
    return cos_sim, pdist

def l2_norm(input):
    input_size = input.size()
    buffer = torch.pow(input, 2)
    normp = torch.sum(buffer, 1).add_(1e-12)
    norm = torch.sqrt(normp)
    _output = torch.div(input, norm.view(-1, 1).expand_as(input))
    output = _output.view(input_size)

    return output

def calc_recall_at_k(T, Y, k):
    """
    T : [nb_samples] (target labels)
    Y : [nb_samples x k] (k predicted labels/neighbours)
    """

    s = 0
    for t,y in zip(T,Y):
        if t in torch.Tensor(y).long()[:k]:
            s += 1
    return s / (1. * len(T))

def _evaluate_cos(X, T):
    # calculate embeddings with model and get targets
    X = l2_norm(X) # 이거 해도 되나?

    # get predictions by assigning nearest 32 neighbors with cosine
    K = 32
    Y = []
    xs = []
    
    cos_sim = F.linear(X, X) # (num of samples) x (num of samples)
    Y = T[cos_sim.topk(1 + K)[1][:,1:]] # select highest similarity sample except itself 
    Y = Y.float().cpu()
    
    recalls = []
    for k in [1, 2, 4, 8, 16, 32]:
        r_at_k = calc_recall_at_k(T, Y, k)
        recalls.append(r_at_k)
        print("R@{} : {:.3f}".format(k, 100 * r_at_k))

    return recalls

def predict_batchwise(model, dataloader):
    """
    :return: list of embeddings and labels
        embeddings: tensor of (num of samples) x (embedding)
        labels: tensor of (num of samples)
    """
    device = "cuda"
    model_is_training = model.training
    model.eval()
    
    ds = dataloader.dataset
    A = [[] for i in range(2)] # A[0]: all embeddings, A[1]: all labels
    with torch.no_grad():
        # extract batches (A becomes list of samples)
        for data, labels, _ in tqdm(dataloader):
            b, _, _, _, _ = data.size()
            data = data.float().cuda("cuda")
            labels = labels.long().view((-1, 1))
            outputs, _, _, _ = model.encode(data)
            for output, label in zip(outputs, labels):
                A[0].append(output)
                A[1].append(label)
    model.train()
    model.train(model_is_training) # revert to previous training state
    
    return [torch.stack(A[i]) for i in range(len(A))] 

def evaluate_one_shot(model, dl_ev, dl_ex):
    query_embeddings, query_labels = predict_batchwise(model, dl_ev) 
    reference_embeddings, reference_labels = predict_batchwise(model, dl_ex)
    embeddings = torch.cat([query_embeddings, reference_embeddings], axis=0)
    labels = torch.cat([query_labels, reference_labels], axis=0)
    recalls = _evaluate_cos(embeddings, labels) # TODO query랑 reference랑 스택
    query_embeddings = l2_norm(query_embeddings) # 이거 해도 되나?
    reference_embeddings = l2_norm(reference_embeddings)
    # https://kevinmusgrave.github.io/pytorch-metric-learning/accuracy_calculation/
    knn_func = CustomKNN(CosineSimilarity())
    knn_distances, knn_indices = knn_func(query_embeddings, 1, reference_embeddings, False)
    #knn_indices, knn_distances = utils.stat_utils.get_knn(reference_embeddings, query_embeddings, 1, False)
    knn_labels = reference_labels[knn_indices][:,0]
    knn_labels_cpu, query_labels_cpu = knn_labels.to('cpu'), query_labels.to('cpu')
    accuracy = accuracy_score(knn_labels_cpu, query_labels_cpu)
    matrix = confusion_matrix(knn_labels_cpu, query_labels_cpu)
    acc_per_class = matrix.diagonal()/matrix.sum(axis=1)
    return recalls, accuracy, embeddings, labels, acc_per_class

def get_cmap(n, name='tab20'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def random_sampling_per_class(embeddings, labels, n_sample=100):
    class_dict = {}
    for i, label in enumerate(labels.reshape(-1)):
        if label not in class_dict:
            class_dict[label] = []
        class_dict[label].append(i)
    sampled_embeddings_list = []
    sampled_labels_list = []
    for label, indice in class_dict.items():
        if len(indice) < n_sample:
            n_sample = len(indice) 
        sampled_indice = random.sample(indice, n_sample)
        sampled_labels_list.append(np.full(n_sample, label))
        sampled_embeddings_list.append(embeddings[sampled_indice])
    
    sampled_labels = np.concatenate(sampled_labels_list)
    sampled_embeddings = np.concatenate(sampled_embeddings_list)
    return sampled_embeddings, sampled_labels

def get_labelnames():
    labelnames = []
    with open('text/pasta_openai_t01.txt') as infile:
        lines = infile.readlines()
        for ind, line in enumerate(lines):
            temp_list = line.rstrip().lstrip().split(';')[0]
            labelnames.append(temp_list)
    return labelnames

def save_tsne_plot(embeddings, labels, root, num_classes=121, epoch=0):
    cmap = get_cmap(num_classes)
    file_name = Path(root) / f"tsne_plot_epoch_{epoch}.png"
    embeddings = embeddings.to('cpu').numpy()
    labels = labels.to('cpu').numpy()
    embeddings, labels = random_sampling_per_class(embeddings, labels, n_sample=200)
    
    tsne_embedded = TSNE(n_components=2, learning_rate='auto',
                        init='random').fit_transform(embeddings)
    
    #plt.title(model_path)
    labelnames = get_labelnames()
    plotted_labels = set()
    for embedding, label in tqdm(zip(tsne_embedded, labels)):
        if label in plotted_labels:
            continue
        plotted_labels.add(label)
        ind = (labels == label)
        selected_embeddings = tsne_embedded[ind]
        xs, ys = selected_embeddings[:,0], selected_embeddings[:,1]
        labelname = f"no{label+1}. {labelnames[label]}"
        plt.scatter(xs, ys, color=cmap(label), s=7, label=labelname)
    plt.legend(loc="center right", bbox_to_anchor=(1.5, 0.5))
    plt.savefig(file_name, bbox_inches='tight')
    plt.clf()


class BalancedSampler(Sampler[int]):

    data_source: Sized
    replacement: bool

    def __init__(self, data_source: Sized, args=None) -> None:
        self.dt = data_source
        self.args = args
        self.n_cls = args.num_class
        self.n_dt = len(self.dt)
        self.n_per_cls = self.dt.n_per_cls
        self.n_cls_wise_desired = int(self.n_dt/self.n_cls)
        self.n_repeat = np.ceil(self.n_cls_wise_desired/np.array(self.n_per_cls)).astype(int)
        self.n_samples = self.n_cls_wise_desired * self.n_cls
        self.st_idx_cls = self.dt.csum_n_per_cls[:-1]
        self.cls_idx = torch.from_numpy(self.st_idx_cls).\
           unsqueeze(1).expand(self.n_cls, self.n_cls_wise_desired)

    def num_samples(self) -> int:
        return self.n_samples

    def __iter__(self):
        batch_rand_perm_lst = list()
        for i_cls in range(self.n_cls):
            rand = torch.rand(self.n_repeat[i_cls], self.n_per_cls[i_cls])
            brp = rand.argsort(dim=-1).reshape(-1)[:self.n_cls_wise_desired]
            batch_rand_perm_lst.append(brp)
        batch_rand_perm  = torch.stack(batch_rand_perm_lst, 0)
        batch_rand_perm += self.cls_idx
        b = batch_rand_perm.permute(1, 0).reshape(-1).tolist()
        yield from b

    def __len__(self):
        return self.num_samples

if __name__ == "__main__":
    create_aligned_dataset()
