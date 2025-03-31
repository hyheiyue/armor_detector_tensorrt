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

#include "armor_detector_tensorrt/tensorrt_detect_node.hpp"

#include "cv_bridge/cv_bridge.h"
#include "fmt/format.h"
#include "rmw/qos_profiles.h"
#include "tf2/LinearMath/Matrix3x3.h"
#include "tf2/LinearMath/Quaternion.h"


#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

namespace rm_auto_aim
{
ArmorDetectorTensorrtNode::ArmorDetectorTensorrtNode(rclcpp::NodeOptions options)
: Node("tensorrt_detect_node", options.use_intra_process_comms(true))
{
  RCLCPP_INFO(this->get_logger(), "Initializing detect node");

  detector_ = nullptr;
  this->initDetector();
  RCLCPP_INFO(this->get_logger(), "Initializing TensorRT");
  if (!detector_) {
    RCLCPP_ERROR(this->get_logger(), "Failed to initialize TensorRT");
    return;
  }
  rcl_interfaces::msg::ParameterDescriptor param_desc;
  param_desc.description = "0-RED, 1-BLUE";
  param_desc.integer_range.resize(1);
  param_desc.integer_range[0].from_value = 0;
  param_desc.integer_range[0].to_value = 1;
  detect_color_ = this->declare_parameter("detect_color", 0, param_desc);

  auto use_sensor_data_qos = this->declare_parameter("use_sensor_data_qos", true);

  camera_name_ = this->declare_parameter("detector.camera_name", "camera");
  detect_color_ = this->declare_parameter("target_color", 0);
  transport_type_ =
    this->declare_parameter("detector.subscribe_compressed", false) ? "compressed" : "raw";
  RCLCPP_INFO(
    this->get_logger(), "camera_name: %s, transport_type: %s", camera_name_.c_str(),
    transport_type_.c_str());

  // Debug mode handler
  RCLCPP_INFO(this->get_logger(), "Setup debug_mode handler");
  debug_mode_ = this->declare_parameter("debug_mode", true);
  if (debug_mode_) {
    this->createDebugPublishers();
 
  }
  // Register debug mode param handler
  // debug_param_sub_ = std::make_shared<rclcpp::ParameterEventHandler>(this);
  // debug_cb_handle_ =
  //   debug_param_sub_->add_parameter_callback("debug_mode", [this](const rclcpp::Parameter & p) {
  //     this->debug_mode_ = p.as_bool();
  //     this->debug_mode_ ? this->createDebugPublishers() : this->destroyDebugPublishers();
  //   });
  param_interface_ = this->get_node_parameters_interface();
  debug_cb_handle_ = param_interface_->add_on_set_parameters_callback(
    [this](const std::vector<rclcpp::Parameter> &params) -> rcl_interfaces::msg::SetParametersResult {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        for (const auto &param : params) {
            if (param.get_name() == "debug_mode") {

                RCLCPP_INFO(this->get_logger(), "Debug mode changed to: %d", param.as_bool());
            }
        }
        return result;
    }
);
  RCLCPP_INFO(this->get_logger(), "Setup ROS subs pubs");
  // Armors publisher
  armors_pub_ = this->create_publisher<auto_aim_interfaces::msg::Armors>(
    "detector/armors", rclcpp::SensorDataQoS());

  // Visualization Marker
  position_marker_.ns = "armors";
  position_marker_.type = visualization_msgs::msg::Marker::SPHERE_LIST;
  position_marker_.scale.x = position_marker_.scale.y = position_marker_.scale.z = 0.1;
  position_marker_.color.a = 1.0;
  position_marker_.color.r = 1.0;

  text_marker_.ns = "classification";
  text_marker_.action = visualization_msgs::msg::Marker::ADD;
  text_marker_.type = visualization_msgs::msg::Marker::TEXT_VIEW_FACING;
  text_marker_.scale.z = 0.1;
  text_marker_.color.a = 1.0;
  text_marker_.color.r = 1.0;
  text_marker_.color.g = 1.0;
  text_marker_.color.b = 1.0;
  marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("detector/marker", 10);

  //Camera handler
  cam_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    camera_name_ + "/camera_info", use_sensor_data_qos ? rclcpp::SensorDataQoS() : rclcpp::QoS(1),
    [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr camera_info) {
      this->cam_info_ = std::make_shared<sensor_msgs::msg::CameraInfo>(*camera_info);
      this->measure_tool_ = std::make_unique<MonoMeasureTool>(
        std::vector<double>(this->cam_info_->k.begin(), this->cam_info_->k.end()),
        this->cam_info_->d);

      RCLCPP_INFO(
        this->get_logger(), "%s",
        fmt::format(
          "Camera intrinsic: {} \ncamera distortion: {}", fmt::join(this->cam_info_->k, " "),
          fmt::join(this->cam_info_->d, " "))
          .c_str());

      // Release subscription
      this->cam_info_sub_.reset();
    });


  img_sub_ = std::make_shared<image_transport::Subscriber>(image_transport::create_subscription(
    this, camera_name_ + "/image",
    std::bind(&ArmorDetectorTensorrtNode::imgCallback, this, std::placeholders::_1), transport_type_,
    use_sensor_data_qos ? rmw_qos_profile_sensor_data : rmw_qos_profile_default));



  RCLCPP_INFO(this->get_logger(), "Subscribing to %s", img_sub_->getTopic().c_str());

  RCLCPP_INFO(this->get_logger(), "Initializing finished.");
}

void ArmorDetectorTensorrtNode::initDetector()
{
  auto model_path = this->declare_parameter("detector.model_path", "");
  
  AdaptedTRTModule::Params params;
  params.input_w = this->declare_parameter("detector.input_width", 416);
  params.input_h = this->declare_parameter("detector.input_height", 416);
  params.num_classes = this->declare_parameter("detector.num_classes", 8);
  params.num_colors = this->declare_parameter("detector.num_colors", 4);
  params.conf_threshold = this->declare_parameter("detector.confidence_threshold", 0.25);
  params.nms_threshold = this->declare_parameter("detector.nms_threshold", 0.3);
  params.top_k = this->declare_parameter("detector.top_k", 128);

  if (model_path.empty()) {
    RCLCPP_ERROR(this->get_logger(), "Require model_path but got empty.");
    return;
  }

  // Create AdaptedTRTModule
  detector_ = std::make_unique<AdaptedTRTModule>(model_path, params);
}

void ArmorDetectorTensorrtNode::imgCallback(const sensor_msgs::msg::Image::ConstSharedPtr  msg)
{
  auto cv_img = cv_bridge::toCvCopy(msg, "rgb8");
  frame_id_ = msg->header.frame_id;
  
  // 直接调用 detect() 进行推理
  const auto objs = detector_->detect(cv_img->image);

  // RCLCPP_INFO(this->get_logger(), "Detected %d armors", bboxes.size());

  // 调用处理函数
  tensorrtDetectCallback(
    objs, msg->header.stamp.sec * 1e9 + msg->header.stamp.nanosec, cv_img->image);
}

void ArmorDetectorTensorrtNode::tensorrtDetectCallback(
  const std::vector<ArmorObject> & objs, int64_t timestamp_nanosec, const cv::Mat & src_img)
{
  if (measure_tool_ == nullptr) {
    RCLCPP_WARN(this->get_logger(), "No camera_info receive yet.");
    return;
  }

  detect_color_ = this->get_parameter("target_color").as_int();

  auto timestamp = rclcpp::Time(timestamp_nanosec);

  // Used to draw debug info
  cv::Mat debug_img;
  if (debug_mode_) {
    debug_img = src_img.clone();
  }

  auto_aim_interfaces::msg::Armors armors_msg;
  armors_msg.header.frame_id = frame_id_;
  armors_msg.header.stamp = timestamp;
  for (auto & obj : objs) {
    if (detect_color_ == 0 && obj.color != ArmorColor::RED) {
      continue;
    } else if (detect_color_ == 1 && obj.color != ArmorColor::BLUE) {
      continue;
    }

    auto_aim_interfaces::msg::Armor armor;

    cv::Point3f target_position;
    cv::Mat target_rvec;
    std::string armor_type;

    if (!measure_tool_->calcArmorTarget(obj, target_position, target_rvec, armor_type)) {
      RCLCPP_WARN(this->get_logger(), "Calc target failed.");
    }

    cv::Mat rot_mat;
    cv::Rodrigues(target_rvec, rot_mat);
    tf2::Matrix3x3 tf_rot_mat(
      rot_mat.at<double>(0, 0), rot_mat.at<double>(0, 1), rot_mat.at<double>(0, 2),
      rot_mat.at<double>(1, 0), rot_mat.at<double>(1, 1), rot_mat.at<double>(1, 2),
      rot_mat.at<double>(2, 0), rot_mat.at<double>(2, 1), rot_mat.at<double>(2, 2));
    tf2::Quaternion tf_quaternion;
    tf_rot_mat.getRotation(tf_quaternion);

    armor.number = K_ARMOR_NAMES[static_cast<int>(obj.number)];
    armor.type = armor_type;
    armor.pose.position.x = target_position.x;
    armor.pose.position.y = target_position.y;
    armor.pose.position.z = target_position.z;
    armor.pose.orientation.x = tf_quaternion.x();
    armor.pose.orientation.y = tf_quaternion.y();
    armor.pose.orientation.z = tf_quaternion.z();
    armor.pose.orientation.w = tf_quaternion.w();
    armor.distance_to_image_center = measure_tool_->calcDistanceToCenter(obj);

    armors_msg.armors.push_back(std::move(armor));

    if (debug_mode_) {
      if (debug_img.empty()) {
        // Avoid debug_mode change in processing
        continue;
      }

      // Draw armor
      for (size_t i = 0; i < 4; ++i) {
        cv::line(debug_img, obj.pts[i], obj.pts[(i + 1) % 4], cv::Scalar(255, 48, 48), 2);
      }

      std::string armor_color;
      switch (obj.color) {
        case ArmorColor::BLUE:
          armor_color = "B";
          break;
        case ArmorColor::RED:
          armor_color = "R";
          break;
        case ArmorColor::NONE:
          armor_color = "N";
          break;
        case ArmorColor::PURPLE:
          armor_color = "P";
          break;
        default:
          armor_color = "UNKNOWN";
          break;
      }

      std::string armor_key = fmt::format("{} {}", armor_color, static_cast<int>(obj.number));
      cv::putText(
        debug_img, armor_key, cv::Point2i(obj.pts[0]), cv::FONT_HERSHEY_SIMPLEX, 0.8,
        cv::Scalar(0, 255, 255), 2);
    }
  }

  armors_pub_->publish(armors_msg);

  if (debug_mode_) {
    if (debug_img.empty()) {
      // Avoid debug_mode change in processing
      return;
    }

    cv::circle(
      debug_img, cv::Point2i(cam_info_->width / 2., cam_info_->height / 2.), 5,
      cv::Scalar(255, 0, 0), 2);

    auto end = this->get_clock()->now();
    auto duration = end.seconds() - timestamp.seconds();
    std::string latency = fmt::format("Latency: {:.3f}ms", duration * 1000);
    cv::putText(
      debug_img, latency, cv::Point2i(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.8,
      cv::Scalar(0, 255, 255), 2);


    debug_img_pub_.publish(cv_bridge::CvImage(armors_msg.header, "rgb8", debug_img).toImageMsg());
  }

}

void ArmorDetectorTensorrtNode::createDebugPublishers()
{
  debug_img_pub_ = image_transport::create_publisher(this, "detector/debug_img");

}

void ArmorDetectorTensorrtNode::destroyDebugPublishers() { debug_img_pub_.shutdown(); }

}  // namespace rm_auto_aim

#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(rm_auto_aim::ArmorDetectorTensorrtNode)
