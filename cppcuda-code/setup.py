import glob
import os.path as osp
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension


ROOT_DIR = osp.dirname(osp.abspath(__file__))
include_dirs = [osp.join(ROOT_DIR, "include")]

sources = glob.glob('*.cpp')+glob.glob('*.cu')


setup(
    name='cppcuda',
    version='0.0',
    author='cpz',
    author_email='@gmail.com',
    description='cppcuda_try',
    long_description='cppcuda_try',
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

























# import glob
# import os.path as ops
# from setuptools import setup
# from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# ROOT_DIR = ops.dirname(ops.abspath(__file__))
# include_dirs = [ops.join(ROOT_DIR, "include")]

# sources=glob.glob('*.cpp')+glob.glob('*.cu')

# setup(
#     name="cppcuda_try",
#     version="1.0",
#     author="cpz",
#     author_email="@",
#     description="cppcuda_try",
#     long_description="cppcuda_try",
#     ext_modules=[
#         CUDAExtension(
#             name="cppcuda_try",
#             sources=sources,
#             include_dirs=include_dirs,
#             # extra_compile_args={"cxx": ["-02"],
#             #                     "nvcc": ['-02']}
#         )
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )



# from setuptools import setup
# from torch.utils.cpp_extension import CppExtension, BuildExtension

# setup(
#     name="cppcuda_try",
#     version="1.0",
#     author="cpz",
#     author_email="@",
#     description="xxx",
#     long_description="This is am example",
#     #需要build出来的源码，可以放多个
#     ext_modules=[
#         CppExtension(
#             name="cppcuda_try",
#             sources=['tt.cpp']
#         ),
#     ],
#     cmdclass={
#         'build_ext': BuildExtension
#     }
# )