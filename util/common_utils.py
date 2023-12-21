import torch
import numpy as np

def norm1_loss(x):
    vec = torch.abs(x)
    return torch.sum(vec)

def norm2_loss(x):
    vec = torch.pow(x, 2)
    return torch.sum(vec)


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
                           np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    else:
        assert False

    return net_input


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False

'''
def psf2otf(psf,size):
    if not(0 in psf): #Pad the PSF to outsize
        psf=np.double(psf)
        psfsize=np.shape(psf)
        psfsize=np.array(psfsize)
        padsize=size-psfsize
        psf=np.lib.pad(psf,((0,padsize[0]),(0,padsize[1])),'constant')
        #Circularly shift otf so that the "center" of the PSF is at the (1,1) element of the array.
        psf=np.roll(psf,-np.array(np.floor(psfsize/2),'i'),axis=(0,1))
        #Compute the OTF
        otf=np.fft.fftn(psf,axes=(0,1))
        #Estimate the rough number of operations involved in the computation of the FFT.
        nElem=np.prod(psfsize,axis=0)
        nOps=0
        for k in range(0,np.ndim(psf)):
            nffts=nElem/psfsize[k]
            nOps=nOps+psfsize[k]*np.log2(psfsize[k])*nffts
        mx1=(abs(np.imag(otf[:])).max(0)).max(0)
        mx2=(abs(otf[:]).max(0)).max(0)
        eps= 2.2204e-16
        if mx1/mx2<=nOps*eps:
            otf=np.real(otf)
    else:
        otf=np.zeros(size)
    return otf
'''
def psf2otf(psf, outSize):
    psfSize = np.array(psf.shape)
    outSize = np.array(outSize)
    padSize = outSize - psfSize
    psf = np.pad(psf, ((0, padSize[0]), (0, padSize[1])), 'constant')
    for i in range(len(psfSize)):
        psf = np.roll(psf, -int(psfSize[i] / 2), i)
    otf = np.fft.fftn(psf)
    nElem = np.prod(psfSize)
    nOps = 0
    for k in range(len(psfSize)):
        nffts = nElem / psfSize[k]
        nOps = nOps + psfSize[k] * np.log2(psfSize[k]) * nffts
    if np.max(np.abs(np.imag(otf))) / np.max(np.abs(otf)) <= nOps * np.finfo(np.float32).eps:
        otf = np.real(otf)
    return otf


def diff3(x,sizeD):
    diff_x = np.zeros([sizeD[0], sizeD[1], sizeD[2], 3, sizeD[3]])
    for i in range(sizeD[3]):
        tenX= x[:,:,i*sizeD[2]:(i+1)*sizeD[2]]
        dfx1 = np.diff(tenX,1,axis=0)
        dfy1 = np.diff(tenX,1,axis=1)
        dfz1 = np.diff(tenX,1,axis=2)
        dfx = np.zeros([sizeD[0], sizeD[1], sizeD[2]])
        dfy = np.zeros([sizeD[0], sizeD[1], sizeD[2]])
        dfz = np.zeros([sizeD[0], sizeD[1], sizeD[2]])
        dfx[0:sizeD[0]-1,:,:]=dfx1
        dfx[sizeD[0]-1,:,:]=tenX[0,:,:]-tenX[sizeD[0]-1,:,:]
        dfy[:,0:sizeD[1] - 1, :] = dfy1
        dfy[:,sizeD[1] - 1, :] = tenX[:, 0, :] - tenX[:, sizeD[1]-1, :]
        dfz[:, :, 0:sizeD[2] - 1] = dfz1
        dfz[:, :, sizeD[2] - 1] = tenX[:, :, 0] - tenX[:, :, sizeD[2]-1]
        diff_x[:, :, :, 0, i] = dfx
        diff_x[:, :, :, 1, i] = dfy
        diff_x[:, :, :, 2, i] = dfz
    return diff_x


def diffT3(a, sizeD):
    tenX = a[:,:,:,0]
    tenY = a[:,:,:,1]
    tenZ = a[:,:,:,2]
    dfx = np.diff(tenX, 1, axis=0)
    dfy = np.diff(tenY, 1, axis=1)
    dfz = np.diff(tenZ, 1, axis=2)
    dfxT = np.zeros([sizeD[0], sizeD[1], sizeD[2]])
    dfyT = np.zeros([sizeD[0], sizeD[1], sizeD[2]])
    dfzT = np.zeros([sizeD[0], sizeD[1], sizeD[2]])
    dfxT[0,:,:] = tenX[sizeD[0]-1,:,:] - tenX[0,:,:]
    dfxT[1:sizeD[0], :, :] = -dfx
    dfyT[:, 0, :] = tenY[:, sizeD[1] - 1, :] - tenY[:, 0, :]
    dfyT[:,1:sizeD[1], :] = -dfy
    dfzT[:, :, 0] = tenZ[:, :, sizeD[2] - 1] - tenZ[:, :, 0]
    dfzT[:, :, 1:sizeD[2]] = -dfz
    diffT_a = dfxT + dfyT + dfzT
    return diffT_a





