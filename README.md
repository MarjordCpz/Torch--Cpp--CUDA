# Torch--Cpp--CUDA
This is my exercise on how to use the C++ bridge to build Pytorch with CUDA. Tutorials from[Pytorch+cpp/cuda extension 教學 - - YouTube](https://www.youtube.com/watch?v=l_Rpk6CRJYI&list=PLDV2CyUo4q-LKuiNltBqCKdO9GH4SS_ec)
## Environment Configuration Description

I first tried to build the setup.py file using VSC+torch 1.12.0-cu116+torchvision 0.13.0-cu116 under Win11. But on Win11 there are always some inexplicable errors. Below is a brief list of errors that occur frequently.

1. Can't find the source file (but the file is clearly under the specified folder)

2. Report a massive and complex error, such as

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

   The solution given on GitHub or stack overflow is

   ```tex
   "not yet, I'm trying, maybe Ubuntu is a kind of solution"
   ```

So I had to run to AutoDL to rent a 2080Ti, the system is ubuntu20.04, and then instantly build it successfully (I won't do deep learning work on Win anymore, it's a lot of work)... Record the detailed build process here.

1. Creating a Virtual Environment

   ```bash
   conda create cppcuda -n python==3.8
   conda activate cppcuda
   (conda init bash)
   pip install torch torchvision
   ```

2. To write a Cpp using VSC, include all the paths in the settings so that you can write it without failing to find the file

   ```json
   "/root/miniconda3/envs/cppcuda/include/python3.8",
   "/root/miniconda3/envs/cppcuda/lib/python3.8/site-packages/torch/include",
   "/root/miniconda3/envs/cppcuda/lib/python3.8/site-packages/torch/include/torch/csrc/api/include"
   ```

## Coding

This implementation is a trilinear interpolation, the algorithm is not too much, is to seek the eigenvalues and then a variety of calculations. Here we explain where parallel processing can be used. We set up two parameters to be processed, one is feature (N, 8, F) and the other point (N, 3), and finally, all the feature points will be calculated out of features_interp (N, F). Obviously, N and F are the parts that need to be processed in parallel. With this processing idea in mind, let's start writing the code.

The general logic of calling each other's code is that Python calls C++ functions via pybind11, and C++ acts as a bridge to implement CUDA's functions in C++. The critical code snippets are as follows.

### cpp

```c++
//Returns the called-out cuda program, which requires the definition of header files
torch::Tensor trilinear_interpolation (
    torch::Tensor feats,
    torch::Tensor points
){
    return trilinear_fw_cu(feats,points); 
}

//Provide Python to call out C++
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation", &trilinear_interpolation);
}
```

Of course, this code omits a few lines of determination code, and the exact definition is written into the header file. Then the next key point is the construction of the cu part.

### cu

I think the overall logic of the cu code is similar to that of the Cpp, which is to define a function and then call it. The difference is that the kernel in cu is called in parallel and needs to specify the number of blocks and threads. Here are some details and templates.

```c++
template <typename scalar_t>      //To ensure consistent data types
__global__ void trilinear_fw_kernel(
    const torch::PackedTensorAccessor<scalar_t, 3, torch::RestrictPtrTraits, size_t> feats,      //feats -> (N, 8, F)
    const torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> points,    //points -> (N, 3)
    torch::PackedTensorAccessor<scalar_t, 2, torch::RestrictPtrTraits, size_t> feat_interp      //feat_interp -> (N, F) Since void, therefore, needs to be passed the result, similar to the double-pointer example.
){
	...
};

torch::Tensor trilinear_fw_cu (
    torch::Tensor feats,
    torch::Tensor points
){
    AT_DISPATCH_FLOATING_TYPES(feats.type(), "trilinear_fw_cu", //Encapsulate the interface, call the kernel function with the same name as this function definition
    ([&]{
        trilinear_fw_kernel<scalar_t><<<blocks, threads>>>(
            feats.packed_accessor<scalar_t, 3, torch::RestrictPtrTraits, size_t>(),  //Data types, dimensions, do not intersect with other variables.
            points.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>(),
            feat_interp.packed_accessor<scalar_t, 2, torch::RestrictPtrTraits, size_t>() 
        );
    }));
    return feat_interp;
}
```

### py

setup.py is used to build out the library, which is then used for testing, hence test.py. Again there are some templates to document:

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

Cd the bash directory to the folder you created, and use the following command to build the library you created.

```bash
pip install .
```

Then use test.py to test the effect and write a Python version of the function that implements the same functionality for time and function testing. The key code is below:

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

The output is as follows, observing that the Cuda code is accelerated by a factor of about 5 (Since this algorithm doesn't use a lot of loops, the speedup isn't significant).

```tex
cuda's cost is: 0.0004818439483642578 s
python's cost is: 0.002289295196533203 s
True
```
