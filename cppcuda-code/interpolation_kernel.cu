#include <torch/extension.h>

template <typename scalar_t>
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,      //feats -> (N, 8, F)
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,    //points -> (N, 3)
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp      //feat_interp -> (N, F)
){
    const int nx = blockIdx.x * blockDim.x + threadIdx.x; 
    const int ny = blockIdx.y * blockDim.y + threadIdx.y;
    if (nx>=feats.size(0) || ny>=feats.size(2)) return;
    //points -> (-1, 1)
    const scalar_t u = (points[nx][0]+1)/2;
    const scalar_t v = (points[nx][1]+1)/2;
    const scalar_t w = (points[nx][2]+1)/2;       

    const scalar_t a = (1-v)*(1-w);
    const scalar_t b = (1-v)*w;
    const scalar_t c = v*(1-w);
    const scalar_t d = 1-a-b-c;
    feat_interp[nx][ny] = (1-u)*(a*feats[nx][0][ny]+
                                 b*feats[nx][1][ny]+
                                 c*feats[nx][2][ny]+
                                 d*feats[nx][3][ny])+
                              u*(a*feats[nx][4][ny]+
                                 b*feats[nx][5][ny]+
                                 c*feats[nx][6][ny]+
                                 d*feats[nx][7][ny]);
};

torch::Tensor trilinear_fw_cu (
    torch::Tensor feats,
    torch::Tensor points
){
    const int N = feats.size(0), F=feats.size(2);
    
    torch::Tensor feat_interp = torch::zeros({N, F}, feats.options());
    // torch::Tensor feat_interp = torch::zeros({N, F}, torch::dtype(torch::kInt32).device(feats.device));
    
    const dim3 threads(16, 16);
    const dim3 blocks((N+threads.x-1)/threads.x, (F+threads.y-1)/threads.y);
    
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", 
    ([&]{
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),  //数据类型， 维度， 不会和其他变量有交集
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>() 
        );
    }));

    return feat_interp;
}
