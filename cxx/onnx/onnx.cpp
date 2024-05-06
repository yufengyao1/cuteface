// onnx.cpp : 此文件包含 "main" 函数。程序执行将在此处开始并结束。
//

#include <iostream>
#include <onnxruntime_cxx_api.h>
int test()
{
     std::cout << "Hello World!\n";

     Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
     Ort::SessionOptions session_options;
     session_options.SetIntraOpNumThreads(1); //设置线程数
     session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
     #ifdef _WIN32
         //const wchar_t *model_path = L"facialexpression.onnx";
         const wchar_t* model_path = L"face.onnx";
     #else
         const char *model_path = "facialexpression.onnx";
     #endif
     Ort::Session session(env, model_path, session_options); //加载模型
     Ort::AllocatorWithDefaultOptions allocator; //内存分配器

     size_t num_input_nodes = session.GetInputCount(); //输入节点1个
     size_t num_output_nodes = session.GetOutputCount(); //输出节点3个

     std::vector<const char *> input_node_names = {"input"};
     std::vector<const char *> output_node_names = {"output","514" ,"513" };
     std::vector<int64_t> input_node_dims = {1,3,44,44}; //input尺寸数组
     size_t input_tensor_size = 3 * 44 * 44; //总尺寸
     std::vector<float> input_tensor_values(input_tensor_size); //图像数据数组
     for (unsigned int i = 0; i < input_tensor_size; i++)
         input_tensor_values[i] = (float)i / (input_tensor_size + 1); //模拟数据
     // create input tensor object from data values
     auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault); 
     Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
     //assert(input_tensor.IsTensor());

     std::vector<Ort::Value> ort_inputs;
     ort_inputs.push_back(std::move(input_tensor));
     //// score model & input tensor, get back output tensor
     auto output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), ort_inputs.data(), input_node_names.size(), output_node_names.data(), output_node_names.size());
     std::cout << "successful!\n";
     return 0;
     //// Get pointer to output tensor float values
     //float *floatarr = output_tensors[0].GetTensorMutableData<float>();
     //float *floatarr_mask = output_tensors[1].GetTensorMutableData<float>();
}

