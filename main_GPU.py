import numpy as np
import math
import torch
import cv2
from matplotlib import pylab as plt
from argparse import ArgumentParser
from numpy.linalg import svd
from PIL import Image
from scipy.io import loadmat, savemat
from util.common_utils import *
from models import *
def psnr3D(img1, img2):
    PIXEL_MAX = 255.0
    PSNR_SUM = 0
    n1,n2,n3=img1.shape
    for i in range(n3):
      mse = np.sum((img1[:, :, i] - img2[:, :, i]) ** 2) / (n1 * n2)
      PSNR_SUM = PSNR_SUM + 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
    return PSNR_SUM / n3

class STLRDP:

    def torch_mul(self, X, Y):  # define (Tensor-Matrix Product))  X: dim1*dim2*r Y:dim3*r return: dim1*dim2*dim3
        dim1, dim2, r = X.shape
        dim3 = Y.shape[0]
        X = self.torch_flattern(X)
        mul = torch.reshape((Y @ X),(dim3, dim1, dim2)).transpose(0,1).transpose(1,2)
        return mul

    def torch_flattern(self,X): # define unrolling tensor along 3-th dimension  X(dim1*dim2*dim3)  -> X(dim3*dim1dim2)
        dim1,dim2,dim3=X.shape
        return torch.reshape(torch.transpose(torch.transpose(X, 2, 1),1,0), (dim3, dim1 * dim2))

    def SoftShrink_GPU(self, X, tau):
        '''
        apply soft thesholding
        '''
        z = torch.sign(X) * (abs(X) - tau) * ((abs(X) - tau) > 0)
        return z

    def tensor_mul(self, a, b):  # 复数点乘，拆成两项  (a+bi)(c+di)=(ac-bd)+(bc+ad)i
        z_real = a.real @ b.real - a.imag @ b.imag
        z_imag = a.real @ b.imag + a.imag @ b.real
        return torch.complex(z_real, z_imag)

    def SVDShrink_Fast_GPU(self, X, tau):
        D = torch.fft.fft(X)
        U_bar = torch.empty(X.shape[0], X.shape[1], 0).to(device)
        S_bar = torch.empty(X.shape[0], X.shape[1], 0).to(device)
        V_bar = torch.empty(X.shape[0], X.shape[1], 0).to(device)
        W_bar = torch.empty((X.shape[0], X.shape[1], 0)).to(device)
        l = X.shape[2]
        for i in range(l):
            if i < math.ceil((l + 1) / 2):
                U, S, V = torch.linalg.svd(D[:, :, i])
                S = self.SoftShrink_GPU(S, tau)
                S = torch.diag(S).to(torch.complex64)
                w = self.tensor_mul(self.tensor_mul(U, S), V)
                # w = torch.dot(torch.dot(U,S),V)
                U_bar = torch.cat((U_bar, U.reshape(X.shape[0], X.shape[1], 1)), 2)
                S_bar = torch.cat((S_bar, S.reshape(X.shape[0], X.shape[1], 1)), 2)
                V_bar = torch.cat((V_bar, V.reshape(X.shape[0], X.shape[1], 1)), 2)
                W_bar = torch.cat((W_bar, w.reshape(X.shape[0], X.shape[1], 1)), 2)
            elif i >= math.ceil((l + 1) / 2):
                U = torch.conj(U_bar[:, :, l - i])
                S = S_bar[:, :, l - i]
                V = torch.conj(V_bar[:, :, l - i])
                w = self.tensor_mul(self.tensor_mul(U, S), V)
                W_bar = torch.cat((W_bar, w.reshape(X.shape[0], X.shape[1], 1)), 2)
        return torch.fft.ifft(W_bar).real

    def ADMM(self,Y,Mask,Mask_Gt,Clean_image,lamada1,lamada2,rank,t,eplison):
        Y=torch.as_tensor(Y).to(device)
        Mask_Gt=torch.tensor(Mask_Gt).to(device).to(torch.float64)
        Mask_4D = torch.transpose(torch.transpose(Mask.to(device), 2, 1), 1, 0).view(1, n3, n1, n2).detach()
        Mask_es_flag = Mask.to(torch.bool)
        Z=torch.zeros(n1,n2,rank).to(device)
        S_new=torch.zeros(n1,n2,n3).to(device)
        Q_new=torch.zeros(n3,rank).to(device).to(torch.float64)
        B_new=torch.zeros(n1,n2,rank).to(device)
        P1= torch.zeros(n1,n2,n3).to(device)
        P2 = torch.zeros(n1,n2,n3).to(device)
        P3= torch.zeros(n1,n2,rank).to(device)
        eplison_mar = eplison * torch.ones(n1, n2).to(device)
        net_input = get_noise(n3, INPUT, (n1, n2)).type(torch.cuda.FloatTensor).detach()
        net = get_net(n3, 'skip', pad, n_channels=n3, skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5,
                      upsample_mode='bilinear').type(torch.cuda.FloatTensor)
        net = net.type(torch.cuda.FloatTensor).to(device)
        X=torch.transpose(torch.transpose(net(net_input).view(n3, n1, n2), 0, 1), 1, 2).detach().to(torch.float64)
        rho1 = rho_R
        rho2 = rho_R
        rho3 = rho_R
        mu_1 = mu
        mu_2 = mu
        mu_3 = mu
        mu_1_max = mu_max
        mu_2_max = mu_max
        mu_3_max = mu_max
        optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
        scheduler =torch.optim.lr_scheduler.StepLR(optimizer, 9000, gamma=0.7, last_epoch=-1)
        PSNR=np.zeros(51)
        m=0
        b=int(n3/t)
        for j in range(num_iter):
            # update Z
            Z_new = self.SVDShrink_Fast_GPU((B_new-(1/mu_3)*P3).to(torch.float32),1/mu_3).to(torch.float64)
            # update S
            S_new = self.SoftShrink_GPU(Y - X + (1/mu_2) * P2, lamada1/mu_2)
            '''             The mask refinement procedure can be skipped
            # update Mask
            S_n=Y - X
            if j >= 10:
                for k in range(t):
                    c = torch.mean(S_n[:, :, k * b:(k + 1) * b], dim=2)
                    A = (c <= eplison_mar).to(torch.int)
                    A = torch.repeat_interleave(A, b, dim=1).view(n1, n2, b)
                    Mask[:, :, k * b:(k + 1) * b] = A
                Mask = Mask.to(torch.float64)
                Mask_es_flag = Mask.to(torch.bool)
            '''
            S_new[Mask_es_flag]=0

            # update B
            B_new = (mu_1 * (torch.matmul(X + (1 / mu_1) * P1, Q_new)) + mu_3 * (Z_new + (1 / mu_3) * P3)) / (
                        mu_1 + mu_3)
            # update Q
            rec=mu_1*(self.torch_flattern(X+(1/mu_1)*P1)) @ (self.torch_flattern(B_new).transpose(0,1))
            U,S_svd,V=torch.svd(rec)
            Q_new=U @ V.transpose(0,1)
            # update theta
            if j < 900:
                inner_iter = 30
            else:
                inner_iter = int(j/30)
            A_net=torch.transpose(torch.transpose(Y-S_new, 2, 1),1,0).view(1,n3,n1,n2).detach()
            B_net=torch.transpose(torch.transpose((torch.matmul(B_new,Q_new.transpose(0,1))-(1/mu_1)*P1), 2, 1),1,0).view(1,n3,n1,n2).detach()
            C_net=torch.transpose(torch.transpose(Y-S_new+(1/mu_2)*P2, 2, 1),1,0).view(1,n3,n1,n2).detach()
            Mask_4D = torch.transpose(torch.transpose(Mask.to(device), 2, 1), 1, 0).view(1, n3, n1, n2).detach()
            for i in range(inner_iter):
                optimizer.zero_grad()
                net_output = net(net_input)
                total_loss = lamada2*norm2_loss(torch.mul(net_output-A_net,Mask_4D))+mu_1/2*norm2_loss(net_output-B_net)
                +mu_2/2*norm2_loss(net_output-C_net)
                # total_loss = norm2_loss(net_output - A_net) + norm2_loss(net_output - B_net)
                total_loss.backward()
                optimizer.step()
                scheduler.step()
            X=torch.transpose(torch.transpose(net(net_input).view(n3, n1, n2), 0, 1), 1, 2).detach().to(torch.float64)
            # update Lagrangian multipliers
            P1=P1+mu_1*(X-self.torch_mul(B_new,Q_new))
            P2=P2+mu_2*(Y-X-S_new)
            P3 =P3+mu_3*(Z_new - B_new)
            mu_1 = min(rho1*mu_1,mu_1_max)
            mu_2 = min(rho2*mu_2,mu_2_max)
            mu_3 = min(rho3*mu_3,mu_3_max)
            if j%show_iter==0:
                X_psnr=X
                X_psnr[Mask_es_flag] = Y[Mask_es_flag]
                StopIe = norm2_loss(X + S_new - Y) / norm2_loss(Y)
                print("iters:", j, "total_loss:", total_loss, "Loss:",lamada2*norm2_loss(torch.mul(net_output-A_net,Mask_4D)),"StopIe:", StopIe)
                print("P1:",torch.sum(P1),"P2:",torch.sum(P2),"P3:",torch.sum(P3))
                Rec_PSNR=psnr3D(X_psnr.detach().cpu().numpy() * 255, np.array(Clean_image * 255))
                print("PSNR:", Rec_PSNR)
                PSNR[m]=Rec_PSNR
                m=m+1
            if j%1000==0:
                data={}
                data.update({'PSNR':PSNR})
                data.update({'X': X_psnr.detach().cpu().numpy() * 255})
                savemat('STLRDP.mat', data)
        rec_img = X_psnr.detach().cpu().numpy()
        rec_cloud=S_new.cpu().numpy()
        return rec_img, rec_cloud

    def Get_Mask(self,Y,lamada1,rank,t,eplison,Mask):
        Y=torch.as_tensor(Y).to(device)
        Z=torch.zeros(n1,n2,rank).to(device)
        S_new=torch.zeros(n1,n2,n3).to(device)
        Q_new=torch.zeros(n3,rank).to(device).to(torch.float64)
        B_new=torch.zeros(n1,n2,rank).to(device)
        P1= torch.zeros(n1,n2,n3).to(device)
        P2 = torch.zeros(n1,n2,n3).to(device)
        P3= torch.zeros(n1,n2,rank).to(device)
        Mask_es=torch.zeros(n1,n2,n3).to(device)
        Mask=torch.tensor(Mask).to(device).to(torch.float64)
        X=torch.zeros(n1,n2,n3).to(torch.float64).to(device)
        rho1 = rho_E
        rho2 = rho_E
        rho3 = rho_E
        mu_1 = mu
        mu_2 = mu
        mu_3 = mu
        mu_1_max = mu_max
        mu_2_max = mu_max
        mu_3_max = mu_max
        b=int(n3/t)
        eplison_mar=eplison*torch.ones(n1,n2).to(device)
        StopIe = norm2_loss(X + S_new - Y) / norm2_loss(Y)
        m=0
        while StopIe>5*1e-6 and m<80:
            # update Z
            Z_new = self.SVDShrink_Fast_GPU((B_new-(1/mu_3)*P3).to(torch.float32),1/mu_3).to(torch.float64)
            # update S
            S_new = self.SoftShrink_GPU(Y - X + (1/mu_2) * P2, lamada1/mu_2)
            # update Mask
            for k in range(t):
                c = torch.mean(S_new[:, :, k * b:(k + 1) * b], dim=2)
                A = (c <= eplison_mar).to(torch.int)
                A = torch.repeat_interleave(A, b, dim=1).view(n1, n2, b)
                Mask_es[:, :, k * b:(k + 1) * b] = A
            Mask_es = Mask_es.to(torch.float64)
            Mask_A = torch.tensor(Mask).cuda().to(torch.int)
            c = torch.sum(abs(Mask_A - Mask_es))
            Mask_es_flag=Mask_es.to(torch.bool)
            S_new[Mask_es_flag]=0
            # update B
            B_new = (mu_1 * (torch.matmul(X + (1 / mu_1) * P1, Q_new)) + mu_3 * (Z_new + (1 / mu_3) * P3)) / (
                        mu_1 + mu_3)
            # update Q
            rec=mu_1*(self.torch_flattern(X+(1/mu_1)*P1)) @ (self.torch_flattern(B_new).transpose(0,1))
            U,S_svd,V=torch.svd(rec)
            Q_new=U @ V.transpose(0,1)
            # update X
            X = (mu_1 * (self.torch_mul(B_new, Q_new) - (1 / mu_1) * P1) + mu_2 * (Y - S_new + (1 / mu_2) * P2)) / (
                        mu_1 + mu_2)
            # update Lagrangian multipliers
            P1=P1+mu_1*(X-self.torch_mul(B_new,Q_new))
            P2=P2+mu_2*(Y-X-S_new)
            P3 =P3+mu_3*(Z_new - B_new)
            mu_1 = min(rho1*mu_1,mu_1_max)
            mu_2 = min(rho2*mu_2,mu_2_max)
            mu_3 = min(rho3*mu_3,mu_3_max)
            m=m+1
            if m%20==0:
                StopIe = norm2_loss(X + S_new - Y) / norm2_loss(Y)
                error=torch.sum(Mask_es-Mask)
                print("m:",m,"Stop_Ie:",StopIe)
                print("err:",error)
        return Mask_es

    def main(self,Cloud_image,Clean_image,Mask,lamada1,lamada2,rank,t,eplison):
        # Estimate Mask
        print("Estimate Mask")
        Mask_es=self.Get_Mask(Cloud_image,lamada1,rank,t,eplison,Mask)
        print("Remove Cloud")
        torch.manual_seed(0)
        rec_img, rec_cloud=self.ADMM(Cloud_image,Mask_es,Mask,Clean_image,lamada1,lamada2,rank,t,eplison)
        return rec_img, rec_cloud



# Load Data
path=r"Cloud Data/Gen_Data1_caseB.mat"
data=loadmat(path)
Clean_image=torch.from_numpy(data['data_clean'])
Cloud_image=torch.from_numpy(data['data_cloud'])
Mask=data['Mask']
Mask = np.array(Mask, dtype= bool)
Mask=~Mask
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Setting Parameters
parser = ArgumentParser(description='STLR-DP')
parser.add_argument('--learning_rate', type=float, default=4*1e-3, help='learning rate')
parser.add_argument('--show_iter', type=float, default=50, help='show the loss during the iteration')
parser.add_argument('--num_iter', type=float, default=1002, help='The total number of iterations')
parser.add_argument('--lambda1', type=float, default=8*1e-5, help='Regularization parameter')
parser.add_argument('--lambda2', type=float, default=20, help='Regularization parameter')
parser.add_argument('--rank', type=float, default=6, help='tucker rank r')
parser.add_argument('--t', type=float, default=5, help='temporal node')
parser.add_argument('--eplison', type=float, default=0.4, help='The threshold for cloud')
parser.add_argument('--mu', type=float, default=1e-3, help='penalty parameter in the ADMM (mu>0)')
parser.add_argument('--mu_max', type=float, default=20, help='maximum of the penalty parameter')
parser.add_argument('--rho_R', type=float, default=1.01, help='update rate of the penalty parameter for cloud removal procedure')
parser.add_argument('--rho_E', type=float, default=1.1, help='update rate of the penalty parameter for Estimating Mask')
args = parser.parse_args()
n1, n2, n3 = Cloud_image.shape
num_iter=args.num_iter
learning_rate=args.learning_rate
lambda1=args.lambda1
lambda2=args.lambda2
rank=args.rank
t=args.t
eplison=args.eplison
show_iter=args.show_iter
mu = args.mu
mu_max = args.mu_max
rho_R = args.rho_R
rho_E = args.rho_E
INPUT = 'noise'
pad = 'reflection'
STLRDP = STLRDP()
B, S= STLRDP.main(Cloud_image,Clean_image,Mask,lambda1,lambda2,rank,t,eplison)

