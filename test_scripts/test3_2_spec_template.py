import pytimeloop.timeloopfe.v4 as tl
from pytimeloop import Config
from pytimeloop.mapping import Mapping
from pytimeloop.model import ArchSpecs
from pytimeloop.problem import Workload

path_arch ='/home/ljy/workspace/mapping/Soter/SpatialAccelerators_v4/Simba/arch.yaml'
path_prob ='/home/ljy/workspace/mapping/Soter/SpatialAccelerators_v4/Simba/problem.yaml'
path_component ='/home/ljy/workspace/mapping/Soter/SpatialAccelerators_v4/Simba/components/*.yaml'
path_map = "/home/ljy/workspace/mapping/Soter/test/v4_soter/map2.yaml"


s1 = tl.Specification.from_yaml_files(path_arch, path_prob, path_component)
print(s1)

####################
arch = tl.arch.Architecture.from_yaml_files(path_arch)
# problem = tl.problem.Problem.from_yaml_files(path_prob)  # 这样也报错!
# s2 = tl.Specification.from_yaml_files(path_map) # 报错, 无architecture
# 1. 读取 mapping YAML 文件
with open(path_map, 'r') as f:
    yaml_str = f.read()
config = Config(yaml_str, "yaml")

# # 2. 需要提供 arch_specs 和 workload
# mapping = Mapping(config.root["mapping"], s1.architecture, s1.problem)
# s2 = tl.Specification(
#     # architecture=arch,
#     # problem=problem,
#     architecture=s1.architecture,
#     problem=s1.problem,
#     mapping=mapping
# )
# print(s2)

# config_base = Config(open(path_arch).read() + open(path_prob).read(), "yaml")
# arch_specs = ArchSpecs(config_base.root["architecture"], False)
# workload = Workload(config_base.root["problem"])
# mapping = Mapping(config.root["mapping"], arch_specs, workload)
#
# # 创建新的 spec
# spec = tl.Specification(
#     architecture=s1.architecture,
#     problem=s1.problem,
#     mapping=mapping
# )
#
# print(spec)

# 1. 预先加载 arch 和 prob 配置
with open(path_arch, 'r') as f:
    arch_yaml = f.read()
with open(path_prob, 'r') as f:
    prob_yaml = f.read()

# 2. 创建 Config 对象
combined_yaml = arch_yaml + "\n" + prob_yaml
config_base = Config(combined_yaml, "yaml")

# 3. 创建 C++ 层的 ArchSpecs 和 Workload
arch_specs = ArchSpecs(config_base.root["architecture"], False)
workload = Workload(config_base.root["problem"])

# 4. 对于每个 mapping 文件
mapping_files = [path_map]
for mapping_file in mapping_files:
    with open(mapping_file, 'r') as f:
        mapping_yaml = f.read()

        # 创建包含 mapping 的 Config
    config_mapping = Config(mapping_yaml, "yaml")

    # 创建 Mapping 对象
    mapping = Mapping(config_mapping.root["mapping"], arch_specs, workload)

# # 1. 预先构建固定的 arch 和 prob
# base_spec = tl.Specification.from_yaml_files(path_arch, path_prob)
# base_spec.process()  # 预处理一次
#
# mapping_files=[...]
# # 2. 对于每个新的 mapping,复制 spec 并更新
# for mapping_file in mapping_files:
#     # # 加载新的 mapping
#     mapping_spec = tl.Specification.from_yaml_files(mapping_file)
#     #
#     # # 复制 base_spec 并更新 mapping
#     # spec = base_spec.copy()  # 或者直接修改
#     # spec.mapping = mapping_spec.mapping
#
#     # 创建新的 spec 实例
#     spec = tl.Specification(
#         architecture=base_spec.architecture,
#         problem=base_spec.problem,
#         mapping=mapping_spec.mapping  # 只更新 mapping
#     )
#     # 评估
#     tl.call_model(spec, output_dir=f"output_{mapping_file}")