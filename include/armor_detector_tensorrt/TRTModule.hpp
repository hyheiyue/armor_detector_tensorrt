#ifndef TRT_MODULE_HPP
#define TRT_MODULE_HPP

#include <memory>
#include <vector>

#include "NvInfer.h"
#include "NvInferRuntime.h"
#include "armor_detector_tensorrt/logging.hpp"
#include "armor_detector_tensorrt/types.hpp"
#include "eigen3/Eigen/Dense"
#include "fmt/color.h"
#include "fmt/core.h"
#include "fmt/printf.h"
#include "opencv2/opencv.hpp"

namespace rm_auto_aim
{
// 定义检测框结构体，与 OpenVINO 模型输出对齐
class AdaptedTRTModule
{
public:
  // 初始化参数结构体
  struct Params
  {
    int input_w = 416;           // 模型输入宽度
    int input_h = 416;           // 模型输入高度
    int num_classes = 8;         // 类别数 (0-7)
    int num_colors = 4;          // 颜色数 (0-3)
    float conf_threshold = 0.3;  // 置信度阈值
    float nms_threshold = 0.5;   // NMS阈值
    int top_k = 128;             // 最大检测框数
  };

  // 构造函数：加载 ONNX 模型并构建 TensorRT 引擎
  explicit AdaptedTRTModule(const std::string & onnx_path, const Params & params);

  // 析构函数：释放资源
  ~AdaptedTRTModule();

  // 推理接口：输入图像，返回检测框列表
  std::vector<ArmorObject> detect(const cv::Mat & image);

private:
  // TensorRT 引擎初始化
  void buildEngine(const std::string & onnx_path);

  // 后处理：解析输出张量，生成检测框
  std::vector<ArmorObject> postprocess(
    std::vector<ArmorObject> & output_objs, std::vector<float> & scores,
    std::vector<cv::Rect> & rects, const float * output, int num_detections,
    const Eigen::Matrix<float, 3, 3> & transform_matrix);

  // 成员变量
  Params params_;
  nvinfer1::ICudaEngine * engine_;
  nvinfer1::IExecutionContext * context_;
  void * device_buffers_[2];  // 输入输出显存指针
  float * output_buffer_;     // 输出数据主机内存
  cudaStream_t stream_;       // CUDA流
  int input_idx_, output_idx_;
  size_t input_sz_, output_sz_;
  // Eigen::Matrix3f transform_matrix; // 变换矩阵
  TRTLogger g_logger_;
};
}  // namespace rm_auto_aim

#endif  // TRT_MODULE_HPP