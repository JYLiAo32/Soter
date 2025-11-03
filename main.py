import os
from datetime import datetime

import yaml
import pickle

import argparse
import numpy as np
import random

from program_tuner import Tuner
from global_config import GlobalConfig


def generate_prob(prob_file: str):
    opt_type = None
    problem = {
        "problem": {
            "shape": {
                "name": "CNN-Layer",
                "dimensions": ["H", "C", "K", "R", "S", "N", "P", "Q"],
                "coefficients": [
                    {"name": "Wstride", "default": 1},
                    {"name": "Hstride", "default": 1},
                    {"name": "Wdilation", "default": 1},
                    {"name": "Hdilation", "default": 1},
                ],
            },
            "instance": {
                "C": 256,
                "K": 512,
                "R": 3,
                "S": 3,
                "P": 56,
                "Q": 56,
                "H": 1,
                "N": 16,
                "Wstride": 1,
                "Hstride": 1,
                "Wdilation": 1,
                "Hdilation": 1,
            },
        }
    }  # TODO: instance也不宜固定

    if args.verbose >= 1:
        print(f"Loading layer problem file and extend it: {prob_file}")

    with open(prob_file, "r") as fd:
        layer_problem = yaml.load(fd, Loader=yaml.SafeLoader)
    if args.verbose >= 2:
        print("\t", layer_problem)

    # 确定prob的类型
    if "type" in layer_problem["problem"].keys() and layer_problem["problem"]["type"] == "T2D":
        # problem["problem"]["instance"]["type"] = "T2D"  # 不再写到文件，新版不支持
        opt_type = "T2D"
        problem["problem"]["shape"]["data_spaces"] = [  # NOTE, 新版的data_spaces是下划线连接的
            {
                "name": "Weights",
                "projection": [[["H"]], [["C"]], [["K"]], [["R"]], [["S"]]],  # NOTE:导出文件会自动被展开成 - - - H 的形式
            },
            {
                "name": "Outputs",
                "projection": [
                    [["N"]],
                    [["H"]],
                    [["K"]],
                    [["R", "Wdilation"], ["P", "Wstride"]],
                    [["S", "Hdilation"], ["Q", "Hstride"]],
                ],
                "read_write": True,
            },
            {
                "name": "Inputs",
                "projection": [[["N"]], [["H"]], [["C"]], [["Q"]], [["P"]]],
            },
        ]
    else:
        # problem["problem"]["instance"]["type"] = "C2D"  # 注意, GEMM也是C2D的一种特例
        opt_type = "C2D"
        problem["problem"]["shape"]["data_spaces"] = [
            {
                "name": "Weights",
                "projection": [[["H"]], [["C"]], [["K"]], [["R"]], [["S"]]],
            },
            {
                "name": "Inputs",
                "projection": [
                    [["N"]],
                    [["H"]],
                    [["C"]],
                    [["R", "Wdilation"], ["P", "Wstride"]],
                    [["S", "Hdilation"], ["Q", "Hstride"]],
                ],
            },
            {
                "name": "Outputs",
                "projection": [[["N"]], [["H"]], [["K"]], [["Q"]], [["P"]]],
                "read_write": True,
            },
        ]

    if "H" in layer_problem["problem"].keys():
        problem["problem"]["instance"]["H"] = layer_problem["problem"]["H"]
    else:
        problem["problem"]["instance"]["H"] = 1  # 保持兼容

    if (
            "type" in layer_problem["problem"].keys()
            and layer_problem["problem"]["type"] == "BMM"
    ):
        problem["problem"]["instance"]["N"] = layer_problem["problem"]["N"]
        problem["problem"]["instance"]["H"] = layer_problem["problem"]["H"] * args.batch_size
    else:
        problem["problem"]["instance"]["N"] = layer_problem["problem"]["N"] * args.batch_size

    problem["problem"]["instance"]["K"] = layer_problem["problem"]["K"]
    problem["problem"]["instance"]["C"] = layer_problem["problem"]["C"]
    problem["problem"]["instance"]["P"] = layer_problem["problem"]["P"]
    problem["problem"]["instance"]["Q"] = layer_problem["problem"]["Q"]
    problem["problem"]["instance"]["R"] = layer_problem["problem"]["R"]
    problem["problem"]["instance"]["S"] = layer_problem["problem"]["S"]
    problem["problem"]["instance"]["Wstride"] = layer_problem["problem"]["Wstride"]
    problem["problem"]["instance"]["Hstride"] = layer_problem["problem"]["Hstride"]
    problem["problem"]["instance"]["Wdilation"] = layer_problem["problem"]["Wdilation"]
    problem["problem"]["instance"]["Hdilation"] = layer_problem["problem"]["Hdilation"]

    return problem, opt_type


def main():
    benchmark_dir = "Benchmarks"
    accelerator_dir = "SpatialAccelerators_v4"
    accelerator = args.accelerator
    workload = args.workload
    layer_id = args.layer_id
    batch_size = args.batch_size

    # 加载workload(problem)
    layer_file = os.path.join(benchmark_dir, f"{workload}_workload/layers.yaml")
    with open(layer_file, "r") as fd:
        if args.verbose >= 2:
            print(f"Loading layer file: {layer_file}")
        layers = yaml.load(fd, Loader=yaml.SafeLoader)
    # 存在间接的文件映射
    layer = layers[layer_id]

    if args.verbose >= 1:
        print("Tuning for:")
        print("\t", accelerator, workload, batch_size, layer_id, layer)

    report_dir = os.path.join(
        args.report_dir,
        f"arch_{accelerator}",
        f"obj_{args.optim_obj}",
        f"{workload}_input{batch_size}",
        f"layer-{layer_id}",
        f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    if args.verbose > 0:
        print(f"Report dir: {report_dir}")
    # 获取确切的简化的problem文件路径
    layer_problem_file = os.path.join(benchmark_dir, f"{workload}_workload/{layer}.yaml")
    # 构建包含完整信息的problem
    problem, opt_type = generate_prob(layer_problem_file)
    # 导出文件
    dump_problem_path = os.path.join(accelerator_dir, accelerator, GlobalConfig.PROBLEM_FILE)
    if args.verbose >= 2:
        print(f"Dumping extended problem file: {dump_problem_path}")
    with open(dump_problem_path, "w") as fd:
        yaml.dump(problem, fd)

    tuner = Tuner(operator_instance=problem["problem"]["instance"],
                  accelerator=accelerator,
                  report_dir=report_dir,
                  optim_obj=args.optim_obj,
                  operator_type=opt_type,
                  verbose=args.verbose,
                  )
    chkpt = tuner.run(args.epochs)
    os.makedirs(report_dir, exist_ok=True)
    dump_env_path = os.path.join(report_dir, GlobalConfig.DUMP_ENV_CKPT_PATH)
    if args.verbose >= 2:
        print(f"Dumping environment checkpoint file: {dump_env_path}")
    with open(dump_env_path, "wb") as fd:
        pickle.dump(chkpt, fd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optim_obj", type=str, default="latency", help="optimization objective")
    parser.add_argument("--epochs", type=int, default=10, help="number of generations/epochs")
    parser.add_argument("--verbose", type=int, default=2, help="logging level")
    parser.add_argument("--report_dir", type=str, default="./report", help="The report directory")

    parser.add_argument("--accelerator", type=str, default="Simba", help="Accelerator name")
    parser.add_argument("--workload", type=str, default=None)
    parser.add_argument("--layer_id", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    ##############################
    # CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj latency --epochs 10 --accelerator Simba --workload resnet50 --layer_id 43 --batch_size 16
    args.optim_obj = "latency"
    args.epochs = 10
    args.accelerator = "Simba"
    args.workload = "resnet50"
    args.layer_id = 43
    args.batch_size = 16
    ##############################

    main()
