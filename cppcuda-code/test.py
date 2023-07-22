import torch
import cppcuda
import time

def trilinear_interpolation_py(feats, points):
    """
    Inputs:
        feats: (N, 8 ,F)
        points: (N, 3) local coordinates in [-1, 1]

    Outputs:
        feats_interp: (N, F)
    """
    u = (points[:, 0:1]+1)/2       # points[:, 0:1] -> (N, 1)
    v = (points[:, 1:2]+1)/2
    w = (points[:, 2:3]+1)/2
    a = (1-w)*(1-v)                # a -> (N, 1)
    b = w*(1-v)
    c = (1-w)*v
    d = 1-a-b-c

    feats_interp = (1-u)*(a*feats[:, 0] +    #feats[:, 0] -> pick a 1 of 8 points' feats -> (N, F)
                          b*feats[:, 1] +
                          c*feats[:, 2] +
                          d*feats[:, 3]) + \
                       u*(a*feats[:, 4] +    
                          b*feats[:, 5] +
                          c*feats[:, 6] +
                          d*feats[:, 7])
    
    return feats_interp



if __name__ == '__main__':
    N = 65536
    F = 256
    feats = torch.rand(N, 8, F, device='cuda')
    points = torch.rand(N, 3, device='cuda')*2-1

    t0=time.time()
    out_cuda = cppcuda.trilinear_interpolation(feats, points)
    print("cuda's cost is: ", time.time()-t0, 's')

    t0=time.time()
    out_py = trilinear_interpolation_py(feats, points)
    print("python's cost is: ", time.time()-t0, 's')
    
    print(torch.allclose(out_py, out_cuda))
    # print(out_py.shape)
    # print(out.shape)