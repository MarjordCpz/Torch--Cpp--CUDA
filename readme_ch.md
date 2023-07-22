# Torch--Cpp--CUDA环境搭建以及简单尝试
教程来源于([Pytorch+cpp/cuda extension 教學 - - YouTube](https://www.youtube.com/watch?v=l_Rpk6CRJYI&list=PLDV2CyUo4q-LKuiNltBqCKdO9GH4SS_ec)

## 环境配置说明

笔者最开始使用Win11下的VSC+torch 1.12.0-cu116+torchvision 0.13.0的版本尝试build出setup.py的文件。但是在Win11上总是出现一些莫名其妙的错误。下面简单列出经常出现的错误。

1. 找不到源文件(但是文件明明就在指定的文件夹下面）

2. 报出一大段复杂的错误，如

   ```powershell
   E:\ProgramData\anaconda3\envs\cppcuda\lib\site-packages\torch\include\pybind11\cast.h(1429): error: too few arguments for template template parameter "Tuple"
               detected during instantiation of class "pybind11::detail::tuple_caster<Tuple, Ts...> [with Tuple=std::pair, Ts=<T1, T2>]"
     (1507): here
     
   E:\ProgramData\anaconda3\envs\cppcuda\lib\site-packages\torch\include\pybind11\cast.h(1503): error: too few arguments for template template parameter "Tuple"
               detected during instantiation of class "pybind11::detail::tuple_caster<Tuple, Ts...> [with Tuple=std::pair, Ts=<T1, T2>]"
     (1507): here
    
    2 errors detected in the compilation of "interpolation_kernel.cu".
     interpolation_kernel.cu
     error: command 'C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.6\\bin\\nvcc.exe' failed with exit code 1
     [end of output]
   ```

   github或stack overflow上面给出的解决方案是

   ```tex
   "not yet,I'm trying, maybe Ubuntu is a kind of solution"
   ```

于是只好跑去AutoDL租用了一块2080Ti，系统是ubuntu20.04，然后瞬间build成功(以后不在Win上做深度学习的工作了，费力不讨好)... 在此记录一下详细的build过程。

1. 创建虚拟环境

   ```bash
   conda create cppcuda -n python==3.8
   conda activate cppcuda
   (conda init bash)
   pip install torch torchvision
   ```

2. 使用VSC编写Cpp，要在设置中把路径全部包含进去，这样写起来才不会出现找不到文件的情况

   ```json
   "/root/miniconda3/envs/cppcuda/include/python3.8",
   "/root/miniconda3/envs/cppcuda/lib/python3.8/site-packages/torch/include",
   "/root/miniconda3/envs/cppcuda/lib/python3.8/site-packages/torch/include/torch/csrc/api/include"
   ```

## 代码编写

本次实现的是三线性插值，算法方面不过多赘述，就是寻求特征值然后各种计算。在此说明一下可以使用到并行处理的地方。我们设定有两个参数需要处理，一个是feats(N, 8, F)另一个是points(N, 3)，最后将所有的特征点全部计算出feats_interp(N, F)。很显然，N和F就是需要并行处理的部分。有了这个处理思路，就来开始编写代码。

代码互相调用的大致逻辑是：python通过pybind11调出C++的函数，在C++的函数种实现CUDA的功能，C++起到了一个桥梁的作用。关键性的代码片段如下。

### cpp部分

```c++
//返回叫出来的cuda程序，需要定义头文件
torch::Tensor trilinear_interpolation (
    torch::Tensor feats,
    torch::Tensor points
){
    return trilinear_fw_cu(feats,points); 
}

//提供python叫出C++的部分
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation", &trilinear_interpolation);
}
```

当然，这段代码省略了几行判定的代码，具体定义就写到了头文件中。那么接下来的关键点就是cu部分的构建。

### cu部分

笔者认为cu代码的整体逻辑和cpp相似，都是定义函数然后在调用。不同的是cu中的这个kernel调用时是并行的需要明确出blocks和threads的数量。下面记录一些细节以及模板。

```c++
template <typename scalar_t>      //为了确保数据类型一致
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,      //feats -> (N, 8, F)
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,    //points -> (N, 3)
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp      //feat_interp -> (N, F) 由于void因此需要传入结果，类似于双指针的例子。
){
	...
};

torch::Tensor trilinear_fw_cu (
    torch::Tensor feats,
    torch::Tensor points
){
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", //封装接口，调出核函数，函数名称和这个函数定义的一致
    ([&]{
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),  //数据类型， 维度， 不会和其他变量有交集
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>() 
        );
    }));
    return feat_interp;
}
```

### py部分

setup.py用来build出来库，再使用这个库进行测试，于是就有了test.py。同样有一些模板要记录：

```python
import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='cppcuda',
    version='0.1',
    author='cpz',
    author_email='@gmail.com',
    description='cppcuda_try',
    long_description='This is an exercise.',
    ext_modules=[
        CUDAExtension(
            name='cppcuda',
            sources=sources,
            include_dirs=include_dirs,
            extra_compile_args={'cxx': ['-O2'],
                                'nvcc': ['-O2']}
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
```

将bash的目录cd到建立的这个文件夹下，使用如下指令build出自己创建的库。

```bash
pip install .
```

再使用test.py测试效果，编写一个python版本的函数实现相同的功能进行时间测试与功能测试。关键代码如下：

```python
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
```

输出结果如下，观察出cuda代码加速了大约5倍(因为这个算法没有用到大量的循环，所以提速并不明显)。

```tex
cuda's cost is: 0.0004818439483642578 s
python's cost is: 0.002289295196533203 s
True
```

