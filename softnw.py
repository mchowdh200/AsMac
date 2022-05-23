import numpy as np
import torch
torch.manual_seed(1)
import torch.nn.functional as F
from torch.autograd import Function
from _softnw import softnw_f_fast, softnw_q_fast, softnw_p_fast, softnw_h_fast, seq2_gradient_fast


def softnw_score(seq1, seq2, gap, gamma=0.01):
    L1 = seq1.shape[1]
    L2 = seq2.shape[1]
    F = np.zeros([L1 + 1, L2 + 1])
    for i in range(L1+1):
        F[i, 0] = -1 * i * gap
    for j in range(1, L2+1):
        F[0, j] = -1 * j * gap
        for i in range(1, L1+1):
            a = F[i-1, j-1] + np.dot(seq1[:, i-1], seq2[:, j-1])
            b = F[i-1, j] - gap
            c = F[i, j-1] - gap
            F[i, j] = gamma_softmax([a, b, c], gamma)

    return F[-1, -1]

def one_hot(s):

    basis = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
    l = len(s)
    feature = np.zeros([4, l])
    for i, c in enumerate(s):
        if c not in ['A', 'T', 'G', 'C']:
            continue
        else:
            feature[basis[c], i] = 1
    return feature


def gamma_softmax(input, gamma=0.01):

    power = [x/gamma for x in input]
    max_power = np.max(power)
    div = [x - max_power for x in power]
    smax = gamma * (max_power + np.log((np.exp(div).sum())))

    return smax


def softnw_f(seq1, seq2, gap, gamma=0.01):
    L1 = seq1.shape[1]
    L2 = seq2.shape[1]
    F = np.zeros([L1 + 2, L2 + 2])
    for i in range(L1+1):
        F[i, 0] = -1 * i * gap
    for j in range(1, L2+1):
        F[0, j] = -1 * j * gap
        for i in range(1, L1+1):
            a = F[i-1, j-1] + np.dot(seq1[:, i-1], seq2[:, j-1])
            b = F[i-1, j] - gap
            c = F[i, j-1] - gap
            F[i, j] = gamma_softmax([a, b, c], gamma)

    return F


def softnw_q(seq1, seq2, F, gap, gamma=0.01):

    L1 = seq1.shape[1]
    L2 = seq2.shape[1]
    seq1 = np.append(seq1, np.zeros([4, 1]), axis=1)
    seq2 = np.append(seq2, np.zeros([4, 1]), axis=1)

    Q = np.zeros([L1 + 2, L2 + 2])
    F[-1, -1] = F[-2, -2]
    Q[-1, -1] = 1

    for i in range(1, L1+1):
        F[i, -1] = 1e8

    for j in [L2 - x for x in range(L2)]:
        F[-1, j] = 1e8
        for i in [L1 - x for x in range(L1)]:
            diff = min(F[i, j] + np.dot(seq1[:, i], seq2[:, j]) - F[i+1, j+1], 0)
            a = np.exp(diff/gamma)
            b = np.exp((F[i, j] - gap - F[i + 1, j]) / gamma)
            c = np.exp((F[i, j] - gap - F[i, j + 1]) / gamma)

            Q[i, j] = a * Q[i+1, j+1] + b * Q[i+1, j] + c * Q[i, j+1]

    return Q

def softnw_p(seq1, seq2, F, gap, gamma=0.01):

    L1 = seq1.shape[1]
    L2 = seq2.shape[1]
    P= np.zeros([L1+2, L2+2])

    for i in range(L1+1):
        P[i, 0] = -1 * i
    for j in range(1, L2+1):
        P[0, j] = -1 * j
        for i in range(1, L1+1):
            diff = (F[i-1, j-1] + np.dot(seq1[:, i - 1], seq2[:, j - 1]) - F[i, j])
            a = np.exp(diff / gamma)
            b = np.exp((F[i-1, j] - gap - F[i, j]) / gamma)
            c = np.exp((F[i, j-1] - gap - F[i, j]) / gamma)
            P[i, j] = a * P[i-1, j-1] + b * (P[i-1, j] - 1) + c * (P[i, j-1] - 1)

    return P[-2, -2]


def softnw_h(seq1, seq2, F):
    L1 = seq1.shape[1]
    L2 = seq2.shape[1]
    H = np.zeros([L1+1, L2+1])
    for j in range(1, L2+1):
        for i in range(1, L1+1):
            H[i, j] = F[i-1, j-1] + np.dot(seq1[:, i-1], seq2[:, j-1]) - F[i, j]

    return H


class SoftNW(Function):

    @staticmethod
    def forward(ctx, seq1, seq2, gap, gamma=0.01):

        gamma = torch.FloatTensor([gamma])
        gap_ = gap.detach().numpy()
        seq1_ = seq1.detach().numpy().astype(np.float64)
        seq2_ = seq2.detach().numpy().astype(np.float64)
        gamma_ = gamma.item()

        # F = torch.FloatTensor(softnw_f(seq1_, seq2_, gap=gap_, gamma=gamma_))
        F = torch.FloatTensor(softnw_f_fast(seq1_, seq2_, gap=gap_, gamma=gamma_))

        ctx.save_for_backward(seq1, seq2, F, gap, gamma)
        return F[-2, -2]

    @staticmethod
    def backward(ctx, grad_output):
        seq1, seq2, F, gap, gamma = ctx.saved_tensors
        F_ = F.detach().numpy().astype(np.float64)
        gap_ = gap.detach().numpy()
        gamma_ = gamma.item()
        seq1_ = seq1.detach().numpy().astype(np.float64)
        seq2_ = seq2.detach().numpy().astype(np.float64)
        output = [None, None, None, None]

        # Q = softnw_q(seq1_, seq2_, F_, gap=gap_, gamma=gamma_)
        Q = softnw_q_fast(seq1_, seq2_, F_, gap=gap_, gamma=gamma_)

        # P = softnw_p(seq1_, seq2_, F_, gap=gap_, gamma=gamma_)
        P = softnw_p_fast(seq1_, seq2_, F_, gap=gap_, gamma=gamma_)

        # H = softnw_h(seq1_, seq2_, F_)
        H = softnw_h_fast(seq1_, seq2_, F_)

        # if seq1.requires_grad:
        #     s1_g = np.zeros(seq1_.shape)
        #     for i in range(1, seq1_.shape[1]+1):
        #         temp_g = np.zeros([1, seq1_.shape[0]])
        #         for j in range(1, seq2_.shape[1]+1):
        #             temp_g += (Q[i, j] * np.exp(H[i, j] / gamma_)) * seq2_[:, j-1]
        #         s1_g[:, i-1] = temp_g
        #     s1_g = torch.FloatTensor(s1_g)
        #     output[0] = grad_output * s1_g

        if seq2.requires_grad:
            # s2_g = np.zeros(seq2_.shape)
            # for j in range(1, seq2_.shape[1]+1):
            #     temp_g = np.zeros([1, seq2_.shape[0]])
            #     for i in range(1, seq1_.shape[1]+1):
            #         temp_g += (Q[i, j] * np.exp(H[i, j] / gamma_)) * seq1_[:, i-1]
            #     s2_g[:, j-1] = temp_g

            s2_g = seq2_gradient_fast(seq1_, seq2_, Q, H, gamma_)
            s2_g = torch.FloatTensor(s2_g)
            output[1] = grad_output * s2_g


        output[2] = grad_output * torch.FloatTensor([P])

        return output[0], output[1], output[2], None
