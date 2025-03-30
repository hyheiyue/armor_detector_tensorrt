* 关于foxy的参数，在yaml中好像不能使用 $(find-pkg-share armor_detector_tensorrt)这种查找路径
* 在launch中components也好像不能加载yaml的参数文件，只能使用内联的字典（不确定是不是我这边环境的问题，找到了类似的issue https://github.com/ros2/launch_ros/issues/156 ，不过看起来正常的情况应该是可以加载yaml），以node的方式launch是完全没问题的，
* 然后就是使用rmw_qos_profile_default在我这边不能对topic回调。
