import os
import pytimeloop.timeloopfe.v4 as tl
from pytimeloop import Config

THIS_SCRIPT_NAME = os.path.basename(__file__).split(".")[0]


# path_arch = "/home/ljy/workspace/mapping/Soter/test/v4_soter/simba_arch.yaml"
# path_prob = "/home/ljy/workspace/mapping/Soter/test/v4_soter/problem.yaml"
path_map = "/home/ljy/workspace/mapping/Soter/test/v4_soter/map.yaml"
# path_component = "/home/ljy/workspace/mapping/Soter/test/v4_soter/components/*.yaml"
path_mapper = "/home/ljy/workspace/mapping/Soter/test/v4_soter/mapper.yaml"

path_arch ='/home/ljy/workspace/mapping/Soter/SpatialAccelerators_v4/Simba/arch.yaml'
path_prob ='/home/ljy/workspace/mapping/Soter/SpatialAccelerators_v4/Simba/problem.yaml'
path_component ='/home/ljy/workspace/mapping/Soter/SpatialAccelerators_v4/Simba/components/*.yaml'
output_path = f"/home/ljy/workspace/mapping/Soter/tmp/output_{THIS_SCRIPT_NAME}"
path_map2 = "/home/ljy/workspace/mapping/Soter/test/v4_soter/map2.yaml"


# spec = tl.Specification.from_yaml_files(path_arch, path_prob, path_component)
# spec = tl.Specification.from_yaml_files(path_arch)

# print(spec)

# print(spec.keys())
# for key in spec.keys():
#     print('------------------')
#     print(key, spec[key])
#     print('------------------')

# print(type(spec["architecture"]))

# arch = spec["architecture"]

# print(arch.keys())
# print(arch.nodes)
# for node in arch.nodes:
#     print(type(node), node)
    

############ 测试extra_input_files用法
# 貌似不行！
# tl.call_mapper(spec, output_dir=output_path, extra_input_files=[path_mapper])
# 与这种做法的效果不同
# spec2 = tl.Specification.from_yaml_files(path_arch, path_prob, path_component, path_mapper)
# tl.call_mapper(spec2, output_dir=output_path)

########### 测试mapping方案
# 不给map.yaml会发生什么？
# tl.call_model(spec, output_dir=output_path)
# 这样行不行？
# tl.call_model(spec, output_dir=output_path, extra_input_files=[path_map])
# 原始做法
# spec3 = tl.Specification.from_yaml_files(path_arch, path_prob, path_component, path_map2)
# tl.call_model(spec3, output_dir=output_path)
# # for key in spec3.keys():
#     print('------------------')
#     print(key, '\n', spec[key], '\n', spec3[key])
#     print('-----------------')

# spec能不能扩充信息？
# 可以！
# spec["mapping"] = tl.Mapping.from_yaml_file(path_map2)
# tl.call_model(spec, output_dir=output_path)


# # 读取 YAML 文件
# with open(path_arch, "r") as f:
#     yaml_str = f.read()
#
# # 使用 Config 类加载
# config = Config(yaml_str, "yaml")
#
# # 访问架构配置
# arch_node = config.root["architecture"]
# print(arch_node)

################
# 单独构建arch\prob
arch = tl.arch.Architecture(version="0.4", nodes=[...])


print('Done!')