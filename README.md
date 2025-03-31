## 相对于humble主要修改了有关参数的加载
* 关于foxy的参数，在yaml中好像不能使用 $(find-pkg-share armor_detector_tensorrt)这种查找路径（foxy没有launch_ros.parameter_descriptions.ParameterFile 的解析方式）-解决方法：在launchfile中设定模型路径
* foxy的compoent不支持类似的
```

armor_detector_tensorrt:
  ros__parameters:
```
的加载方式，issue https://github.com/ros2/launch_ros/issues/156 ,这里的解决方法独立了两种启动方法的yaml，以不同的形式描述

  
* 然后就是使用rmw_qos_profile_default在我这边不能对topic回调，需要use_sensor_qos
