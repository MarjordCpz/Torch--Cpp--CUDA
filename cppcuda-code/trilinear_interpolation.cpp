// #include <stdio.h>
#include <torch/extension.h>
#include "utils.h"


torch::Tensor trilinear_interpolation (
    torch::Tensor feats,
    torch::Tensor points
){
    CHECK_INPUT(feats);
    CHECK_INPUT(points);
    return trilinear_fw_cu(feats,points); //返回叫出来的cuda程序，需要定义头文件
    // return feats;
}

//提供python叫出C++的部分
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("trilinear_interpolation", &trilinear_interpolation);
}