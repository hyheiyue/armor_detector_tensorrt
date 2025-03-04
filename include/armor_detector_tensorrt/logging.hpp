// Copyright 2025 Zikang Xie
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ARMOR_DETECTOR_TENSORRT__LOGGING_HPP_
#define ARMOR_DETECTOR_TENSORRT__LOGGING_HPP_

#include <iostream>

#include "NvInfer.h"

class TRTLogger : public nvinfer1::ILogger
{
public:
  explicit TRTLogger(nvinfer1::ILogger::Severity severity = nvinfer1::ILogger::Severity::kWARNING)
  : severity_(severity)
  {
  }
  void log(nvinfer1::ILogger::Severity severity, const char * msg) noexcept override
  {
    if (severity <= severity_) {
      std::cerr << msg << std::endl;
    }
  }
  nvinfer1::ILogger::Severity severity_;
};

#endif  // ARMOR_DETECTOR_TENSORRT__LOGGING_HPP_
