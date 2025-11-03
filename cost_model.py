import numpy as np
import yaml
import os, sys
import copy
import math
from functools import reduce
from collections import defaultdict, OrderedDict
from subprocess import Popen, PIPE, call
import logging
import pathlib
import re
from concurrent.futures import ProcessPoolExecutor
import pytimeloop.timeloopfe.v4 as tl
from pytimeloop.timeloopfe.common.nodes import DictNode, ListNode
from global_config import GlobalConfig
from pathlib import Path
import joblib
from utils.util import redirect_output_to_file


def to_plain_obj(obj):
    """递归地将 DictNode/ListNode 转换为纯 Python dict/list"""
    if isinstance(obj, dict):
        return {k: to_plain_obj(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_plain_obj(item) for item in obj]
    else:
        return obj


# TODO: 大改此文件！适配新版timeloop

###################
# 对外接口：
#  1. 若干属性的读取接口
#  2. run函数: 应该就是批量program评估
#  3. thread_fun: 应该就是每个program单独评估，

class Timeloop:
    def __init__(self, in_config_path: str,
                 out_config_path: str,
                 accelerator: str,
                 opt_obj: list = None, use_sparse: bool = False, verbose: int = 1):
        self.input_config_path = Path(in_config_path) / accelerator
        self.accelerator = accelerator
        self.out_config_path = out_config_path
        self.use_sparse = use_sparse
        self.opt_obj = opt_obj
        self.verbose = verbose

        # 仅包含芯片架构与问题配置文件
        self.base_spec = tl.Specification.from_yaml_files(
            self.input_config_path / GlobalConfig.PROBLEM_FILE,
            self.input_config_path / GlobalConfig.ARCHITECTURE_FILE,
            self.input_config_path / GlobalConfig.COMPONENT_PATH,
        )
        self.architecture = self.base_spec.architecture
        # self.problem = dict(self.base_spec.problem)  # 从tl的类实例转为dict
        self.problem = to_plain_obj(self.base_spec.problem)  # 从tl的类实例转为dict

        # 解析架构信息：缓冲区名称、大小、空间映射约束等
        buffer_name_list, buffer_size_list, buffer_spmap_cstr, num_buffer_levels, num_pes, constrain_dict = (
            self.get_arch_info())
        self.buffer_name_list = buffer_name_list
        self.buffer_size_list = buffer_size_list
        self.buffer_spmap_cstr = buffer_spmap_cstr  # 各层级的spatial划分上限(受buffer size限制)
        self.buffers_with_spmap = set(
            [key for key, value in self.buffer_spmap_cstr.items() if value > 1])  # FIXME: 感觉不对？buffer并行关系怎么应对
        self.num_buffer_level = num_buffer_levels
        self.num_pes = num_pes
        self.constrain_dict = constrain_dict

        # 默认缓冲区能耗成本（单位：pJ/access）
        self.buf_energy_cost = self.get_default_buffer_energy_cost()

        # 稀疏密度配置：输入/权重/输出的稀疏程度
        self.density = {'Inputs': 0.5, 'Weights': 1, 'Outputs': 1}

        # R=S=卷积核尺寸, P=Q=输出特征图尺寸, C=输入通道, K=输出通道, H=输入高度, N=批大小
        # 固定各维度的顺序编码
        self.dim2note = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'H', 7: 'N'}
        if self.verbose >= 1:
            print('dim2note:', self.dim2note.values())

        # 获取问题维度信息和质因数分解
        self.dimension, self.dimension_dict = self.get_problem_info()
        self.dimension_prime = {key: self.get_prime_factors(self.dimension_dict[key]) for key in self.dim2note.values()}

        # 构建质因数到索引的映射，用于编码优化策略
        self.prime2idx = {}
        primes = set()
        for i, key in enumerate(self.dim2note.values()):
            tile_budget = self.dimension_prime[key]
            for k in tile_budget.keys():
                primes.add(int(k))
        primes = sorted(primes)  # 所有维度中出现的质因数的集合
        self.prime2idx = {'{}'.format(pf): i for i, pf in enumerate(primes)}  # 对出现的质因数按照从小到大排序
        self.num_primes = len(self.prime2idx.keys())

        # 初始化配置文件路径列表（用于并行执行）
        self.arch_path, self.problem_path, self.map_path, self.sparse_path, self.pool_path = [], [], [], [], []

    def get_default_buffer_energy_cost(self):
        """
        获取默认的缓冲区访问能耗成本
        单位：pJ/access (皮焦耳每次访问)

        返回:
            dict: 各存储级别的能耗成本字典
        """
        buf_energy_cost = {'DRAM': 200,  # DRAM访问能耗   # FIXME: 是否需要修正或补充?
                           'l2': 2.2,  # L2缓存访问能耗
                           'l1': 1.12,  # L1缓存访问能耗
                           'MAC': 1.0,  # MAC单元计算能耗
                           }
        return buf_energy_cost

    def get_num_buffer_levels(self):
        """获取存储层次结构的层数"""
        return self.num_buffer_level

    def get_buffers_with_spmap(self):
        """获取支持空间映射的缓冲区集合"""
        return self.buffers_with_spmap

    def get_buffer_spmap_cstr(self):
        """获取各缓冲区的空间映射约束"""
        return self.buffer_spmap_cstr

    def get_buffer_size_list(self):
        """获取各缓冲区的大小列表"""
        return self.buffer_size_list

    def get_prime_factors(self, n):
        """
        计算一个数的质因数分解
        例如：12 = 2^2 * 3^1

        参数:
            n: 待分解的整数

        返回:
            defaultdict: 质因数及其幂次的字典
        """
        primes = defaultdict(int)
        # 处理因子2
        while n % 2 == 0:
            primes['2'] += 1
            n = n // 2
        # 处理奇数因子
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            while n % i == 0:
                primes[f'{i}'] += 1
                n = n // i
        # 处理剩余的质数
        if n > 2:
            primes[f'{n}'] += 1
        return primes

    def get_factors(self, n):
        """
           找出整数 n 的所有约数
            两两组合，返回所有约数的列表
        """
        return list(reduce(list.__add__,
                           ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

    def get_dimension_factors(self, dimension_dict):
        """
        获取所有问题维度的因子列表

        参数:
            dimension_dict: 问题维度字典

        返回:
            dict: 各维度因子的字典
        """
        dimension_factors = dict()
        for key, value in dimension_dict.items():
            factors = self.get_factors(value)
            dimension_factors[key] = factors
        return dimension_factors

    def get_dimension_primes(self):
        """
        获取问题维度的质因数分解信息

        返回:
            tuple: (维度列表, 维度质因数字典, 质因数到索引映射)
        """
        return self.dimension, self.dimension_prime, self.prime2idx

    def get_problem_info(self):
        """
        从问题配置文件中提取维度信息
        """
        dimension = []
        dimension_dicts = {}
        for key in self.dim2note.values():
            value = self.problem['instance'][key]
            dimension.append(value)
            dimension_dicts[key] = value
        return dimension, dimension_dicts

    def get_arch_info(self):
        """
        解析架构配置文件，提取硬件资源信息
        支持多种加速器架构：Simba、Eyeriss、TensorCore

        返回:
            tuple: (缓冲区名称字典, 缓冲区大小字典, 空间映射约束字典, 缓冲区层数, PE数量)
        """
        arch = self.architecture
        buffer_name_list = []
        buffer_size_list = []
        num_instances = []
        num_buffer_levels = 0
        constrain_dict = dict()  # 记录每个内存层级的约束
        # TODO: 提取信息, 参照旧版做法
        # 1. 遍历arch.nodes
        # 2. 判断类型:
        #   - container: 可能定义公共信息(如spatial)
        #   - parallel: 存在子nodes; 可能定义公共信息(如spatial)
        #   - computer/macc: 提取目标
        # 3. 按顺序提取信息, 确保结果正确
        # 4. parallel部分的提取要谨慎！
        if self.accelerator == 'Simba':
            # 顺序: System -> DRAM -> Chip -> GlobalBuffer -> PE ->
            # Parallel(InputBuffer, WeightBuffer, AccumulationBuffer) ->
            # reg_mac -> Registers -> MACC
            def extract_buffer_info(node):
                buffer_name = node['name']
                attributes = node['attributes']
                depth = attributes.get('depth', None)
                depth = depth if depth is not None else float('Inf')
                data_width = attributes.get('datawidth', 8)
                width = attributes.get('width', 8)
                block_size = attributes.get('block_size', 1)
                assert width == data_width * block_size, 'width must be equal to data_width * block_size'
                buffer_size = depth * block_size
                instance = 1
                if 'spatial' in node:
                    meshX = node['spatial'].get('meshX', 1)
                    meshY = node['spatial'].get('meshY', 1)
                    instance *= meshX * meshY

                # NOTE: 仅提取需要的约束
                constraint = node['constraints']['dataspace']
                constraint = {k: to_plain_obj(v) for k, v in constraint.items() if
                              k in ['keep', 'bypass', 'type', 'target']}
                constraint['target'] = buffer_name
                return buffer_name, buffer_size, instance, constraint

            #############################
            # 主存
            main_memory = arch.nodes[1]
            buffer_name, buffer_size, instance, constraint = extract_buffer_info(main_memory)
            buffer_name_list.append(buffer_name)
            num_instances.append(instance)
            buffer_size_list.append(buffer_size)
            constrain_dict[buffer_name] = constraint
            num_buffer_levels += 1
            #############################
            # Chip
            # 跳过, 无可用信息
            #############################
            # GlobalBuffer
            global_buffer = arch.nodes[3]
            buffer_name, buffer_size, instance, constraint = extract_buffer_info(global_buffer)
            buffer_name_list.append(buffer_name)
            num_instances.append(instance)
            buffer_size_list.append(buffer_size)
            constrain_dict[buffer_name] = constraint
            num_buffer_levels += 1
            #############################
            # PE
            pe = arch.nodes[4]
            pe_instance = 1  # 记录并行数量
            if 'spatial' in pe:
                meshX = pe['spatial'].get('meshX', 1)
                meshY = pe['spatial'].get('meshY', 1)
                pe_instance *= meshX * meshY
            ## Parallel
            ### InputBuffer, WeightBuffer, AccumulationBuffer
            parallel_component = arch.nodes[5]
            for buffer in parallel_component.nodes:
                buffer_name, buffer_size, instance, constraint = extract_buffer_info(buffer)
                buffer_name_list.append(buffer_name)
                num_instances.append(instance * pe_instance)  # 属于PE的内部组件, 需要乘以PE的数量！
                buffer_size_list.append(buffer_size)
                constrain_dict[buffer_name] = constraint
                num_buffer_levels += 1  # FIXME: 并行层级, level可能需要分层处理
            ###################
            ## reg_mac
            reg_mac = arch.nodes[6]
            reg_mac_instance = 1 * pe_instance  # 属于PE的内部组件, 需要乘以PE的数量！
            if 'spatial' in reg_mac:
                meshX = reg_mac['spatial'].get('meshX', 1)
                meshY = reg_mac['spatial'].get('meshY', 1)
                reg_mac_instance *= meshX * meshY
            ### Registers
            reg = arch.nodes[7]
            buffer_name, buffer_size, reg_instance, constraint = extract_buffer_info(reg)
            constrain_dict[buffer_name] = constraint
            reg_instance *= reg_mac_instance  # 属于reg_mac的内部组件, 需要乘以reg_mac的数量！
            buffer_name_list.append(buffer_name)
            num_instances.append(reg_instance)
            buffer_size_list.append(buffer_size)
            num_buffer_levels += 1

            ### MACC
            macc = arch.nodes[8]
            # macc_instance = 1 * reg_mac_instance  # 属于reg_mac的内部组件, 需要乘以reg_mac的数量！
            macc_instance = 1 * reg_instance  # FIXME: 似乎应该算作属于reg! 但一般而言两者数量相同
            if 'spatial' in macc:
                meshX = macc['spatial'].get('meshX', 1)
                meshY = macc['spatial'].get('meshY', 1)
                macc_instance *= meshX * meshY
            num_instances.append(macc_instance)  # 属于reg_mac的内部组件, 需要乘以reg_mac的数量！

        elif 'Eyeriss' in self.accelerator:
            pass
        elif 'TensorCore' in self.accelerator:
            pass

        # if self.verbose >= 1:
        #     print(buffer_name_list, num_instances, buffer_size_list, constrain_dict)

        # 计算空间映射约束 - 每个缓冲层允许的最大空间映射因子
        sp_cstr = []
        for i in range(len(num_instances) - 1):
            # FIXME: 后续可能需要改成树状的分层计算, 因为InputBuffer, WeightBuffer, AccumulationBuffer是同级别的关系
            allowed_sp_size = num_instances[i + 1] // num_instances[i]
            sp_cstr.append(allowed_sp_size)
            if num_instances[i + 1] % num_instances[i] != 0:
                raise ValueError('Invalid Architecture File. '
                                 'Buffer hierarchy not perfectly divisible.')
        if self.verbose >= 1:
            print(sp_cstr)

        # 返回重构后的字典格式，按存储层次编号（l1, l2, l3...）
        return {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_name_list)}, \
            {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_size_list)}, \
            {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), sp_cstr)}, \
            num_buffer_levels, macc_instance, constrain_dict

    def get_dimension_dict(self, dim_value):
        """
        将维度值数组转换为维度名字典

        参数:
            dim_value: 维度值数组

        返回:
            dict: 维度名到值的映射字典
        """
        return {note: value for note, value in zip(self.dim2note.values(), dim_value)}

    def get_tp_sp_tile_size(self, dim_value, sp_dim, sp_dim_value, timeloop_notation=True):
        """
        将维度值分解为时间映射和空间映射的瓦片大小

        参数:
            dim_value: 维度值字典
            sp_dim: 空间映射的维度集合
            sp_dim_value: 空间映射的维度值字典
            timeloop_notation: 是否使用Timeloop格式输出

        返回:
            tuple: (时间映射字符串, 空间映射字符串) 或 (时间映射数组, 空间映射数组)
        """
        if timeloop_notation:
            temporal_series = []
            spatial_series = []
            for note, value in dim_value.items():
                if note not in sp_dim:
                    temporal_series.append(f'{note}={value}')  # 不并行, 则所有分块同属于temporal
                    spatial_series.append(f'{note}=1')
                else:
                    sp_value = sp_dim_value[note]
                    tp_value = value // sp_value  # 并行, 则temporal的数量等于总分块数量除以并行数量
                    temporal_series.append(f'{note}={tp_value}')
                    spatial_series.append(f'{note}={sp_value}')
            return ' '.join(temporal_series), ' '.join(spatial_series)
            # temporal_series = f'[{",".join(temporal_series)}]'
            # spatial_series = f'[{",".join(spatial_series)}]'
            # return temporal_series, spatial_series
        else:
            temporal_series = []
            spatial_series = []
            for note in self.dim2note.values():
                if note not in sp_dim:
                    temporal_series.append(dim_value[note])
                    spatial_series.append(1)
                else:
                    sp_value = sp_dim_value[note]
                    tp_value = dim_value[note] // sp_value
                    temporal_series.append(tp_value)
                    spatial_series.append(sp_value)
            return np.array(temporal_series), np.array(spatial_series)

    def create_pool_env(self, num_pools):
        """
        为并行执行创建独立的工作环境
        每个进程池有独立的目录和配置文件

        参数:
            num_pools: 进程池数量
        """
        os.makedirs(self.out_config_path, exist_ok=True)
        arch_paths, problem_paths, map_paths, sparse_paths, pool_paths = [], [], [], [], []
        for i in range(num_pools):
            pool_dir = os.path.join(self.out_config_path, f'pool-{i}')
            os.makedirs(pool_dir, exist_ok=True)
            pool_paths.append(pool_dir)
            arch_paths.append(os.path.abspath(os.path.join(pool_dir, 'arch.yaml')))
            problem_paths.append(os.path.abspath(os.path.join(pool_dir, 'problem.yaml')))
            map_paths.append(os.path.abspath(os.path.join(pool_dir, 'map.yaml')))
            sparse_paths.append(os.path.abspath(os.path.join(pool_dir, 'sparse.yaml')))
        self.arch_path, self.problem_path, self.map_path, self.sparse_path, self.pool_path = arch_paths, problem_paths, map_paths, sparse_paths, pool_paths

    def get_problem_configs(self, dimension):
        """
        根据给定的维度生成问题配置文件
        支持稀疏计算配置

        参数:
            dimension: 维度值列表

        返回:
            dict: 完整的问题配置字典
        """
        problem = copy.deepcopy(self.problem)
        dimension_dict = self.get_dimension_dict(dimension)
        for key, value in dimension_dict.items():
            problem['instance'][key] = value
        if self.use_sparse:
            problem['instance']['densities'] = {}
            for key in ['Inputs', 'Weights', 'Outputs']:
                cur_density = self.density[key]
                if cur_density < 1:
                    problem['instance']['densities'][key] = {}
                    problem['instance']['densities'][key]['distribution'] = 'fixed-structured'
                    # problem['instance']['densities'][key]['distribution'] = 'hypergeometric'
                    problem['instance']['densities'][key]['density'] = cur_density
        problem = {'problem': problem}
        return problem

    def get_map_config(self, program):
        # FIXME: 临时修正方案
        redirect_spatial_target = {
            'GlobalBuffer': 'PE',
            'AccumulationBuffer': 'reg_mac'
        }
        additional_map_info = [
            {
                'target': 'PE',
                'type': 'dataspace',
                'keep': [],
                'bypass': ['Weights', 'Inputs', 'Outputs'],
            },
            {
                'target': 'PE',
                'type': 'temporal',
                'factors': 'H=1 C=1 K=1 R=1 S=1 N=1 P=1 Q=1',
                'permutation': 'HCKRSNPQ',
            },
            {
                'target': 'reg_mac',
                'type': 'dataspace',
                'keep': [],
                'bypass': ['Weights', 'Inputs', 'Outputs'],
            },
            {
                'target': 'reg_mac',
                'type': 'temporal',
                'factors': 'H=1 C=1 K=1 R=1 S=1 N=1 P=1 Q=1',
                'permutation': 'HCKRSNPQ',
            }
        ]

        steps_per_level = len(self.dim2note.values())
        mapping = []
        # self.check_tile_fit_buffer(program)
        num_primes = len(self.prime2idx.keys())
        for level in range(1, self.num_buffer_level + 1):
            target = self.buffer_name_list[f'l{level}']
            level_program = program[(level - 1) * steps_per_level:level * steps_per_level, :]
            par_dims = set()
            perm_list = copy.deepcopy(list(self.dim2note.values()))
            tile_sizes_dict = {}
            sp_tile_sizes_dict = {}
            # program中每维代表什么意思？
            #   0: 维度的索引, 对应dim2note
            #   1~num_primes(2): 对应prime2idx的系数, 应该是描述该维度的分块因子与每个prime的关系
            #   num_primes + 1(3): 当前维度的并行因子, 为1时表示2不并行
            #   4: 什么意思？怎么没用？
            for i in range(steps_per_level):
                # note = dim2note[level_program[i, 0]]
                order = level_program[i, 0]
                note = self.dim2note[i]
                perm_list[order] = note
                if level_program[i, num_primes + 1] >= 1:  # 当前维度并行因子大于等于1, 表示并行
                    par_dims.add(note)  # 记录当前内存层级存在并行的维度
                tile_sizes_dict[note] = 1
                for k, v in self.prime2idx.items():  # 通过这段代码可以印证, 从中计算出当前维度的分块因子
                    tile_sizes_dict[note] *= pow(int(k), level_program[i, int(v) + 1])
                sp_tile_sizes_dict[note] = pow(2, level_program[i, num_primes + 1])  # FIXME: 并行因子的幂默认取2

            # permutation = []
            # for i in range(steps_per_level):
            #     permutation.append(perm_list[i])
            # permutation = f'[{",".join(permutation)}]'

            permutation = ''
            for i in range(steps_per_level):
                permutation += perm_list[i]

            # print(perm_list)
            # bypass_map = self.mapspace['mapspace']['constraints'][level - 1]
            bypass_map = self.constrain_dict[target]

            tp_tile_sizes, sp_tile_sizes = self.get_tp_sp_tile_size(tile_sizes_dict, par_dims, sp_tile_sizes_dict)

            cur_map = {'target': target,
                       'type': 'temporal',
                       'factors': tp_tile_sizes,
                       'permutation': permutation,
                       }
            mapping.append(cur_map)
            if f'l{level}' in self.buffers_with_spmap:
                cur_map = {'target': target,
                           'type': 'spatial',
                           'factors': sp_tile_sizes,
                           'permutation': permutation,
                           }
                # FIXME: 临时修正方案
                assert target in redirect_spatial_target.keys()
                cur_map['target'] = redirect_spatial_target[target]
                mapping.append(cur_map)
            mapping.append(bypass_map)
        # FIXME: 临时修正方案
        mapping += additional_map_info
        return {'mapping': mapping}

    def get_configs(self, dimension, program):
        map = self.get_map_config(program)
        return arch, problem, map

    def write_config(self, arch, problem, map, arch_path, problem_path, map_path, sparse_path=None):
        # with open(arch_path, 'w') as fd:  # 沿用原文件
        #     yaml.dump(arch, fd)
        # with open(problem_path, 'w') as fd:  # TODO_D: 沿用原文件，但是先对比程序是否有修改它; 没有修改
        #     yaml.dump(problem, fd)
        with open(map_path, 'w') as fd:
            yaml.dump(map, fd)
        if self.use_sparse:
            with open(sparse_path, 'w') as fd:
                yaml.dump(self.sparse, fd)

    def judge(self, stats, opt_obj='all'):
        if opt_obj == 'all':
            opt_obj = ['edp', 'latency', 'energy']
        ret = []

        for f in opt_obj:
            if f == 'edp':
                ret.append(-stats['cycles'] * stats['energy'])  # energy_uJ
            if f == 'latency':
                ret.append(-stats['cycles'])
            # if f == 'utilization':
            #     ret.append(stats['utilization'])
            if f == 'energy':
                ret.append(-stats['energy'])  # energy_uJ
        return ret

    def run_config(self, filename):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # capture everything

        # Output file names.
        out_prefix = "timeloop-model."
        report_prefix = out_prefix + 'stats.txt'
        xml_file_name = out_prefix + "map+stats.xml"

        filename = pathlib.Path(filename).resolve()
        report_file = filename / report_prefix
        status_dict = dict()
        if report_file.exists():  # FIXME: 这个文件不存在，大概是路径问题
            with open(report_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                m = re.match(r"Energy: (.*) uJ", line)
                if m:
                    energy = m.group(1)
                    status_dict['energy'] = float(energy)
                else:
                    # m = re.match(r"Max topology cycles: (.*)", line)
                    m = re.match(r"Cycles: (.*)", line)
                    if m:
                        cycle = m.group(1)
                        status_dict['cycles'] = int(cycle)
        return status_dict

    def dump_map_config(self, program, file_path):
        map = self.get_map_config(program)
        base_dir = os.path.dirname(file_path)
        os.makedirs(base_dir, exist_ok=True)
        with open(file_path, 'w') as fd:
            yaml.dump(map, fd)

    # 基于新版timeloopPE接口的评估方法
    def thread_fun(self, program, pool_idx, output_path: str = None):
        if output_path is None:
            map_file_path = self.map_path[pool_idx]
        else:
            map_file_path = os.path.join(output_path, GlobalConfig.MAP_PATH)
        base_dir = os.path.dirname(map_file_path)
        os.makedirs(base_dir, exist_ok=True)

        arch_file_path = self.input_config_path / GlobalConfig.ARCHITECTURE_FILE
        comp_file_path = self.input_config_path / GlobalConfig.COMPONENT_PATH
        prob_file_path = self.input_config_path / GlobalConfig.PROBLEM_FILE

        output_path = os.path.join(base_dir, "model")
        spec_log_path = os.path.join(base_dir, "spec_log.txt")
        model_log_path = os.path.join(base_dir, "model_log.txt")

        try:
            self.dump_map_config(program, map_file_path)
            with redirect_output_to_file(spec_log_path):
                spec = tl.Specification.from_yaml_files(
                    arch_file_path,
                    comp_file_path,
                    prob_file_path,
                    map_file_path,
                )

            res = tl.call_model(spec, output_dir=output_path, log_to=model_log_path)

            cycle = res.cycles
            energy = res.energy * 1e6  # uJ, 保持单位与输出文件的summary部分一致
            # energy = res.per_compute("energy") * 1e12
            stats = {
                'cycles': cycle,
                'energy': energy,
            }
            fitness = self.judge(stats, self.opt_obj)
        except Exception as e:
            print("Exception: ", e)
            fitness = [-float('Inf')] * len(self.opt_obj)
        return fitness

    def run_old(self, programs):
        # test: programs.shape: (32, 48, 5),
        # 分别指(batch_size, 决策长度(6个内存层级*8个循环维度), 每维的信息(维度索引、位置编码、分块因子、并行因子、？)
        num_samples = programs.shape[0]
        # pool = ProcessPoolExecutor(num_samples)
        pool = None
        self.create_pool_env(num_pools=num_samples)

        fitness = np.ones((num_samples, len(self.opt_obj))) * -np.inf

        if not pool:
            for i, program in enumerate(programs):
                fit = self.thread_fun(program, 0)
                fitness[i] = fit
        else:
            while (1):
                try:
                    fits = list(pool.map(self.thread_fun, zip(programs, np.arange(len(programs)))))
                    for i, fit in enumerate(fits):
                        fitness[i] = fit
                    break
                except Exception as e:
                    print(type(e).__name__, e)
                    pool.shutdown(wait=False)
                    pool = ProcessPoolExecutor(num_samples)

        return fitness

    def run(self, programs, parallel_run: bool = True):
        # test: programs.shape: (32, 48, 5),
        # 分别指(batch_size, 决策长度(6个内存层级*8个循环维度), 每维的信息(维度索引、位置编码、分块因子、并行因子、？)
        num_samples = programs.shape[0]
        self.create_pool_env(num_pools=num_samples)
        fitness = np.ones((num_samples, len(self.opt_obj))) * -np.inf

        if not parallel_run:
            for i, program in enumerate(programs):
                fit = self.thread_fun(program, 0)
                fitness[i] = fit
        else:
            # n_jobs=None: 单线程, -1: 所有CPU核心, 其他正值: 指定线程数
            fitness = joblib.Parallel(n_jobs=-1)(
                joblib.delayed(self.thread_fun)(program, i) for i, program in enumerate(programs)
            )
            fitness = np.array(fitness)
        return fitness


class TimeloopOld(object):
    """
    Timeloop成本模型主类
    用于评估深度学习加速器在不同映射策略下的性能表现
    """

    def __init__(self, in_config_path='./SpatialAccelerators', out_config_path='./out_config', accelerator='Simba',
                 opt_obj=None, use_sparse=False, verbose=1):
        """
        初始化Timeloop成本模型

        参数:
            in_config_path: 输入配置文件路径，包含架构、问题、映射空间等YAML文件
            out_config_path: 输出配置文件路径，用于存放生成的配置文件
            accelerator: 加速器类型，如'Simba'、'Eyeriss'、'TensorCore'等
            opt_obj: 优化目标列表，如['edp', 'latency', 'energy']
            use_sparse: 是否使用稀疏计算优化
        """

        self.accelerator = accelerator
        self.out_config_path = out_config_path
        self.use_sparse = use_sparse

        # 加载架构配置文件 - 定义硬件加速器结构
        print(os.path.join(in_config_path, accelerator, 'arch.yaml'))
        with open(os.path.join(in_config_path, accelerator, 'arch.yaml'), 'r') as fd:
            self.arch = yaml.load(fd, Loader=yaml.SafeLoader)

        # 加载问题配置文件 - 定义神经网络层参数
        with open(os.path.join(in_config_path, accelerator, 'problem.yaml'), 'r') as fd:
            self.problem = yaml.load(fd, Loader=yaml.SafeLoader)

        # 加载映射空间配置文件 - 定义合法的映射策略空间
        with open(os.path.join(in_config_path, accelerator, 'mapspace.yaml'), 'r') as fd:
            self.mapspace = yaml.load(fd, Loader=yaml.SafeLoader)

        self.opt_obj = opt_obj

        # 如果使用稀疏优化，加载稀疏配置文件
        if self.use_sparse:
            with open(os.path.join(in_config_path, accelerator, 'sparse.yaml'), 'r') as fd:
                self.sparse = yaml.load(fd, Loader=yaml.SafeLoader)

        # 解析架构信息：缓冲区名称、大小、空间映射约束等
        buffer_name_list, buffer_size_list, buffer_spmap_cstr, num_buffer_levels, num_pes = self.get_arch_info()
        self.buffer_name_list = buffer_name_list
        self.buffer_size_list = buffer_size_list
        self.buffer_spmap_cstr = buffer_spmap_cstr
        self.buffers_with_spmap = set([key for key, value in self.buffer_spmap_cstr.items() if value > 1])
        self.num_buffer_level = num_buffer_levels
        self.num_pes = num_pes

        # Timeloop可执行文件名
        self._executable = 'timeloop-model'

        # 默认缓冲区能耗成本（单位：pJ/access）
        self.buf_energy_cost = self.get_default_buffer_energy_cost()

        # 稀疏密度配置：输入/权重/输出的稀疏程度
        self.density = {'Inputs': 0.5, 'Weights': 1, 'Outputs': 1}
        # self.dim_note = ['R', 'S', 'P', 'Q', 'C', 'K', 'H', 'N']
        # self.dim2note = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'N', 7: 'H'}
        # 维度映射：将数字索引映射到问题维度名称
        # R=S=卷积核尺寸, P=Q=输出特征图尺寸, C=输入通道, K=输出通道, H=输入高度, N=批大小
        self.dim2note = {0: 'R', 1: 'S', 2: 'P', 3: 'Q', 4: 'C', 5: 'K', 6: 'H', 7: 'N'}
        print(self.dim2note.values())

        # 获取问题维度信息和质因数分解
        self.dimension, self.dimension_dict = self.get_problem_info()
        self.dimension_prime = {key: self.get_prime_factors(self.dimension_dict[key]) for key in self.dim2note.values()}

        # 构建质因数到索引的映射，用于编码优化策略
        self.prime2idx = {}
        primes = set()
        for i, key in enumerate(self.dim2note.values()):
            tile_budget = self.dimension_prime[key]
            for k in tile_budget.keys():
                primes.add(int(k))
        primes = sorted(primes)
        self.prime2idx = {'{}'.format(pf): i for i, pf in enumerate(primes)}
        self.num_primes = len(self.prime2idx.keys())

        # 初始化配置文件路径列表（用于并行执行）
        self.arch_path, self.problem_path, self.map_path, self.sparse_path, self.pool_path = [], [], [], [], []

    def get_default_buffer_energy_cost(self):
        """
        获取默认的缓冲区访问能耗成本
        单位：pJ/access (皮焦耳每次访问)

        返回:
            dict: 各存储级别的能耗成本字典
        """
        buf_energy_cost = {'DRAM': 200,  # DRAM访问能耗
                           'l2': 2.2,  # L2缓存访问能耗
                           'l1': 1.12,  # L1缓存访问能耗
                           'MAC': 1.0,  # MAC单元计算能耗
                           }
        return buf_energy_cost

    def get_num_buffer_levels(self):
        """获取存储层次结构的层数"""
        return self.num_buffer_level

    def get_buffers_with_spmap(self):
        """获取支持空间映射的缓冲区集合"""
        return self.buffers_with_spmap

    def get_buffer_spmap_cstr(self):
        """获取各缓冲区的空间映射约束"""
        return self.buffer_spmap_cstr

    def get_buffer_size_list(self):
        """获取各缓冲区的大小列表"""
        return self.buffer_size_list

    def get_prime_factors(self, n):
        """
        计算一个数的质因数分解
        例如：12 = 2^2 * 3^1

        参数:
            n: 待分解的整数

        返回:
            defaultdict: 质因数及其幂次的字典
        """
        primes = defaultdict(int)
        # 处理因子2
        while n % 2 == 0:
            primes['2'] += 1
            n = n // 2
        # 处理奇数因子
        for i in range(3, int(math.sqrt(n)) + 1, 2):
            while n % i == 0:
                primes[f'{i}'] += 1
                n = n // i
        # 处理剩余的质数
        if n > 2:
            primes[f'{n}'] += 1
        return primes

    def get_factors(self, n):
        """
        获取一个数的所有因子

        参数:
            n: 待求因子的整数

        返回:
            list: 所有因子的列表
        """
        return list(reduce(list.__add__,
                           ([i, n // i] for i in range(1, int(n ** 0.5) + 1) if n % i == 0)))

    def get_dimension_factors(self, dimension_dict):
        """
        获取所有问题维度的因子列表

        参数:
            dimension_dict: 问题维度字典

        返回:
            dict: 各维度因子的字典
        """
        dimension_factors = dict()
        for key, value in dimension_dict.items():
            factors = self.get_factors(value)
            dimension_factors[key] = factors
        return dimension_factors

    def get_dimension_primes(self):
        """
        获取问题维度的质因数分解信息

        返回:
            tuple: (维度列表, 维度质因数字典, 质因数到索引映射)
        """
        return self.dimension, self.dimension_prime, self.prime2idx

    def get_problem_info(self):
        """
        从问题配置文件中提取维度信息

        返回:
            tuple: (维度值列表, 维度名字典)
        """
        problem = copy.deepcopy(self.problem)
        dimension = []
        dimension_dicts = {}
        for key in self.dim2note.values():
            value = problem['problem']['instance'][key]
            dimension.append(value)
            dimension_dicts[key] = value
        return dimension, dimension_dicts

    def get_arch_info(self):
        """
        解析架构配置文件，提取硬件资源信息
        支持多种加速器架构：Simba、Eyeriss、TensorCore

        返回:
            tuple: (缓冲区名称字典, 缓冲区大小字典, 空间映射约束字典, 缓冲区层数, PE数量)
        """
        arch = copy.deepcopy(self.arch)
        buffer_name_list = []
        buffer_size_list = []
        num_instances = []
        num_buffer_levels = 0
        arch = arch['architecture']

        if self.accelerator == 'Simba':
            # Simba架构解析 - 三层存储结构：DRAM -> Global Buffer -> PE Local
            main_memory = arch['subtree'][0]['local'][0]
            buffer_name = main_memory['name']
            attributes = main_memory['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:  # FIXME_D: 这里是干嘛的？ 取出并行的数量及名称，例如从PE[0..15]提取信息
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            global_buffer = arch['subtree'][0]['subtree'][0]['local'][0]
            buffer_name = global_buffer['name']
            attributes = global_buffer['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size  # NOTE: width = word_bits(datawidth) * block_size, 这里depth*block_size是全局缓冲区的大小(可存放data的数量)
            instances = 1
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            pe = arch['subtree'][0]['subtree'][0]['subtree'][0]
            num_pes = int(pe['name'].split('..')[1].split(']')[0]) + 1

            for buf in pe['local'][:-1]:  # 最后一个是MACC, 不是存储单元
                buffer_name = buf['name']
                attributes = buf['attributes']
                depth = attributes['depth'] if 'depth' in attributes else float('Inf')
                word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
                width = attributes['width'] if 'width' in attributes else 8
                block_size = attributes['block-size'] if 'block-size' in attributes else 1
                buffer_size = depth * block_size
                instances = 1
                re_ret = re.search('.*\[', buffer_name)
                if re_ret:
                    instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                    buffer_name = re_ret.group(0)[:-1]
                instances *= num_pes
                buffer_name_list.append(buffer_name)
                buffer_size_list.append(buffer_size)
                num_instances.append(instances)
                num_buffer_levels += 1

            macc = pe['local'][-1]['name']
            instances = int(macc.split('..')[1].split(']')[0]) + 1
            instances *= num_pes
            num_instances.append(instances)
        elif 'Eyeriss' in self.accelerator:
            main_memory = arch['subtree'][0]['local'][0]
            buffer_name = main_memory['name']
            attributes = main_memory['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            global_buffer = arch['subtree'][0]['subtree'][0]['local'][0]
            buffer_name = global_buffer['name']
            attributes = global_buffer['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            dummy_buffer = arch['subtree'][0]['subtree'][0]['local'][1]
            buffer_name = dummy_buffer['name']
            attributes = dummy_buffer['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                buffer_name = re_ret.group(0)[:-1]
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            pe = arch['subtree'][0]['subtree'][0]['subtree'][0]
            num_pes = int(pe['name'].split('..')[1].split(']')[0]) + 1

            for buf in pe['local'][:-1]:
                buffer_name = buf['name']
                attributes = buf['attributes']
                depth = attributes['depth'] if 'depth' in attributes else float('Inf')
                word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
                width = attributes['width'] if 'width' in attributes else 8
                block_size = attributes['block-size'] if 'block-size' in attributes else 1
                buffer_size = depth * block_size
                instances = 1
                re_ret = re.search('.*\[', buffer_name)
                if re_ret:
                    instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
                    buffer_name = re_ret.group(0)[:-1]
                instances *= num_pes
                buffer_name_list.append(buffer_name)
                buffer_size_list.append(buffer_size)
                num_instances.append(instances)
                num_buffer_levels += 1

            macc = pe['local'][-1]['name']
            re_ret = re.search('.*\[', macc)
            if re_ret:
                instances = (int(macc.split('..')[1].split(']')[0]) + 1) * num_pes
            else:
                instances = num_pes
            num_instances.append(instances)
        elif 'TensorCore' in self.accelerator:
            main_memory = arch['subtree'][0]
            buffer_name = main_memory['name']
            attributes = main_memory['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            instances = 1
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances = int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = main_memory['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            global_buffer = arch['subtree'][0]['subtree'][0]
            buffer_name = global_buffer['name']
            attributes = global_buffer['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = global_buffer['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            local_buffer = arch['subtree'][0]['subtree'][0]['subtree'][0]
            buffer_name = local_buffer['name']
            attributes = local_buffer['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = local_buffer['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1

            pe_buffer = arch['subtree'][0]['subtree'][0]['subtree'][0]['subtree'][0]
            buffer_name = pe_buffer['name']
            attributes = pe_buffer['local'][0]['attributes']
            depth = attributes['depth'] if 'depth' in attributes else float('Inf')
            word_bits = attributes['word-bits'] if 'word-bits' in attributes else 8
            width = attributes['width'] if 'width' in attributes else 8
            block_size = attributes['block-size'] if 'block-size' in attributes else 1
            buffer_size = depth * block_size
            re_ret = re.search('.*\[', buffer_name)
            if re_ret:
                instances *= int(buffer_name.split('..')[1].split(']')[0]) + 1
            buffer_name = pe_buffer['local'][0]['name']
            buffer_name_list.append(buffer_name)
            buffer_size_list.append(buffer_size)
            num_instances.append(instances)
            num_buffer_levels += 1
            num_pes = instances

            macc = arch['subtree'][0]['subtree'][0]['subtree'][0]['subtree'][0]['local'][1]['name']
            re_ret = re.search('.*\[', macc)
            if re_ret:
                instances *= (int(macc.split('..')[1].split(']')[0]) + 1)
            num_instances.append(instances)

        print(buffer_name_list, num_instances, buffer_size_list)

        # 计算空间映射约束 - 每个缓冲层允许的最大空间映射因子
        sp_cstr = []
        for i in range(len(num_instances) - 1):
            allowed_sp_size = num_instances[i + 1] // num_instances[i]
            sp_cstr.append(allowed_sp_size)
            if num_instances[i + 1] % num_instances[i] != 0:
                raise ValueError('Invalid Architecture File. '
                                 'Buffer hierarchy not perfectly divisible.')

        print(sp_cstr)

        # 返回重构后的字典格式，按存储层次编号（l1, l2, l3...）
        return {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_name_list)}, \
            {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), buffer_size_list)}, \
            {f'l{level}': name for level, name in zip(np.arange(num_buffer_levels, 0, -1), sp_cstr)}, \
            num_buffer_levels, num_pes

    def get_dimension_dict(self, dim_value):
        """
        将维度值数组转换为维度名字典

        参数:
            dim_value: 维度值数组

        返回:
            dict: 维度名到值的映射字典
        """
        return {note: value for note, value in zip(self.dim2note.values(), dim_value)}

    def get_tp_sp_tile_size(self, dim_value, sp_dim, sp_dim_value, timeloop_notation=True):
        """
        将维度值分解为时间映射和空间映射的瓦片大小

        参数:
            dim_value: 维度值字典
            sp_dim: 空间映射的维度集合
            sp_dim_value: 空间映射的维度值字典
            timeloop_notation: 是否使用Timeloop格式输出

        返回:
            tuple: (时间映射字符串, 空间映射字符串) 或 (时间映射数组, 空间映射数组)
        """
        if timeloop_notation:
            temporal_series = []
            spatial_series = []
            for note, value in dim_value.items():
                if note not in sp_dim:
                    temporal_series.append(f'{note}={value}')
                    spatial_series.append(f'{note}=1')
                else:
                    sp_value = sp_dim_value[note]
                    tp_value = value // sp_value
                    temporal_series.append(f'{note}={tp_value}')
                    spatial_series.append(f'{note}={sp_value}')

            return ' '.join(temporal_series), ' '.join(spatial_series)
        else:
            temporal_series = []
            spatial_series = []
            for note in self.dim2note.values():
                if note not in sp_dim:
                    temporal_series.append(dim_value[note])
                    spatial_series.append(1)
                else:
                    sp_value = sp_dim_value[note]
                    tp_value = dim_value[note] // sp_value
                    temporal_series.append(tp_value)
                    spatial_series.append(sp_value)
            return np.array(temporal_series), np.array(spatial_series)

    def create_pool_env(self, num_pools):
        """
        为并行执行创建独立的工作环境
        每个进程池有独立的目录和配置文件

        参数:
            num_pools: 进程池数量
        """
        os.makedirs(self.out_config_path, exist_ok=True)
        arch_paths, problem_paths, map_paths, sparse_paths, pool_paths = [], [], [], [], []
        for i in range(num_pools):
            pool_dir = os.path.join(self.out_config_path, f'pool-{i}')
            os.makedirs(pool_dir, exist_ok=True)
            pool_paths.append(pool_dir)
            arch_paths.append(os.path.abspath(os.path.join(pool_dir, 'arch.yaml')))
            problem_paths.append(os.path.abspath(os.path.join(pool_dir, 'problem.yaml')))
            map_paths.append(os.path.abspath(os.path.join(pool_dir, 'map.yaml')))
            sparse_paths.append(os.path.abspath(os.path.join(pool_dir, 'sparse.yaml')))
        self.arch_path, self.problem_path, self.map_path, self.sparse_path, self.pool_path = arch_paths, problem_paths, map_paths, sparse_paths, pool_paths

    def get_problem_configs(self, dimension):
        """
        根据给定的维度生成问题配置文件
        支持稀疏计算配置

        参数:
            dimension: 维度值列表

        返回:
            dict: 完整的问题配置字典
        """
        problem = copy.deepcopy(self.problem)
        dimension_dict = self.get_dimension_dict(dimension)
        for key, value in dimension_dict.items():
            problem['problem']['instance'][key] = value
        if self.use_sparse:
            problem['problem']['instance']['densities'] = {}
            for key in ['Inputs', 'Weights', 'Outputs']:
                cur_density = self.density[key]
                if cur_density < 1:
                    problem['problem']['instance']['densities'][key] = {}
                    problem['problem']['instance']['densities'][key]['distribution'] = 'fixed-structured'
                    # problem['problem']['instance']['densities'][key]['distribution'] = 'hypergeometric'
                    problem['problem']['instance']['densities'][key]['density'] = cur_density
        return problem

    def get_map_config(self, program):
        steps_per_level = len(self.dim2note.values())
        mapping = []
        # self.check_tile_fit_buffer(program)
        num_primes = len(self.prime2idx.keys())
        for level in range(1, self.num_buffer_level + 1):
            target = self.buffer_name_list[f'l{level}']
            level_program = program[(level - 1) * steps_per_level:level * steps_per_level, :]
            par_dims = set()
            perm_list = copy.deepcopy(list(self.dim2note.values()))
            tile_sizes_dict = {}
            sp_tile_sizes_dict = {}
            for i in range(steps_per_level):
                # note = dim2note[level_program[i, 0]]
                order = level_program[i, 0]
                note = self.dim2note[i]
                perm_list[order] = note
                if level_program[i, num_primes + 1] >= 1:
                    par_dims.add(note)
                tile_sizes_dict[note] = 1
                for k, v in self.prime2idx.items():
                    tile_sizes_dict[note] *= pow(int(k), level_program[i, int(v) + 1])
                sp_tile_sizes_dict[note] = pow(2, level_program[i, num_primes + 1])

            permutation = ''
            for i in range(steps_per_level):
                permutation += perm_list[i]

            # print(perm_list)
            bypass_map = self.mapspace['mapspace']['constraints'][level - 1]
            tp_tile_sizes, sp_tile_sizes = self.get_tp_sp_tile_size(tile_sizes_dict, par_dims, sp_tile_sizes_dict)

            cur_map = {'target': target,
                       'type': 'temporal',
                       'factors': tp_tile_sizes,
                       'permutation': permutation,
                       }
            mapping.append(cur_map)
            if f'l{level}' in self.buffers_with_spmap:
                cur_map = {'target': target,
                           'type': 'spatial',
                           'factors': sp_tile_sizes,
                           'permutation': permutation,
                           }
                mapping.append(cur_map)
            mapping.append(bypass_map)
        return {'mapping': mapping}

    def get_configs(self, dimension, program):
        arch = self.arch
        problem = self.get_problem_configs(dimension)
        map = self.get_map_config(program)
        return arch, problem, map

    def write_config(self, arch, problem, map, arch_path, problem_path, map_path, sparse_path=None):
        with open(arch_path, 'w') as fd:
            yaml.dump(arch, fd)
        with open(problem_path, 'w') as fd:
            yaml.dump(problem, fd)
        with open(map_path, 'w') as fd:
            yaml.dump(map, fd)
        if self.use_sparse:
            with open(sparse_path, 'w') as fd:
                yaml.dump(self.sparse, fd)

    def thread_fun(self, program, pool_idx):
        arch, problem, map = self.get_configs(self.dimension, program)
        self.write_config(arch, problem, map, arch_path=self.arch_path[pool_idx],
                          problem_path=self.problem_path[pool_idx], map_path=self.map_path[pool_idx],
                          sparse_path=self.sparse_path[pool_idx])
        command = [self._executable, self.arch_path[pool_idx], self.problem_path[pool_idx], self.map_path[pool_idx]]
        if self.use_sparse:
            command += [self.sparse_path[pool_idx]]
        process = Popen(command, stdout=PIPE, stderr=PIPE, cwd=self.pool_path[pool_idx])
        stdout, stderr = process.communicate()
        process.wait()
        if stderr:
            print("stderrstderr: ", stderr, program)
            return [-float('Inf')] * len(self.opt_obj)
        else:
            try:
                stats = self.run_config(self.pool_path[pool_idx])
                fitness = self.judge(stats, self.opt_obj)
            except Exception as e:
                print("Exception: ", e)
                fitness = [-float('Inf')] * len(self.opt_obj)
            return fitness

    def run(self, programs):
        num_samples = programs.shape[0]
        # pool = ProcessPoolExecutor(num_samples)
        pool = None
        self.create_pool_env(num_pools=num_samples)

        fitness = np.ones((num_samples, len(self.opt_obj))) * -np.inf

        if not pool:
            for i, program in enumerate(programs):
                fit = self.thread_fun((program, 0))
                fitness[i] = fit
        else:
            while (1):
                try:
                    fits = list(pool.map(self.thread_fun, zip(programs, np.arange(len(programs)))))
                    for i, fit in enumerate(fits):
                        fitness[i] = fit
                    break
                except Exception as e:
                    print(type(e).__name__, e)
                    pool.shutdown(wait=False)
                    pool = ProcessPoolExecutor(num_samples)

        return fitness

    def judge(self, stats, opt_obj='all'):
        if opt_obj == 'all':
            opt_obj = ['edp', 'latency', 'energy']
        ret = []

        for f in opt_obj:
            if f == 'edp':
                ret.append(-stats['cycles'] * stats['energy'])  # energy_uJ
            if f == 'latency':
                ret.append(-stats['cycles'])
            # if f == 'utilization':
            #     ret.append(stats['utilization'])
            if f == 'energy':
                ret.append(-stats['energy'])  # energy_uJ
        return ret

    def run_config(self, filename):
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)  # capture everything

        # Output file names.
        out_prefix = "timeloop-model."
        report_prefix = out_prefix + 'stats.txt'
        xml_file_name = out_prefix + "map+stats.xml"

        filename = pathlib.Path(filename).resolve()
        report_file = filename / report_prefix
        status_dict = dict()
        if report_file.exists():  # FIXME: 这个文件不存在，大概是路径问题
            with open(report_file, 'r') as f:
                lines = f.readlines()
            for line in lines:
                m = re.match(r"Energy: (.*) uJ", line)
                if m:
                    energy = m.group(1)
                    status_dict['energy'] = float(energy)
                else:
                    # m = re.match(r"Max topology cycles: (.*)", line)
                    m = re.match(r"Cycles: (.*)", line)
                    if m:
                        cycle = m.group(1)
                        status_dict['cycles'] = int(cycle)
        return status_dict


if __name__ == '__main__':
    model = TimeloopOld()
