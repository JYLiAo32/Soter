import os
import yaml
import pickle

import argparse
import numpy as np
import random

from program_tuner import Tuner


def main():
    benchmark_dir = 'Benchmarks'
    accelerator_dir = 'SpatialAccelerators'
    accelerator = args.accelerator
    workload = args.workload
    layer_id = args.layer_id
    batch_size = args.batch_size

    layer_file = os.path.join(benchmark_dir, '{}_workload/layers.yaml'.format(workload))
    with open(layer_file, 'r') as fd:
        if args.verbose >= 2:
            print('Loading layer file: {}'.format(layer_file))
        layers = yaml.load(fd, Loader=yaml.SafeLoader)

    layer = layers[layer_id]
    
    if args.verbose >= 1:
        print('Tuning for:')
        print('\t', accelerator, workload, batch_size, layer_id, layer)
    
    report_dir = os.path.join(args.report_dir,  'arch_{}'.format(accelerator), 'obj_{}'.format(args.optim_obj),
                              '{}_input{}'.format(workload, batch_size), 'layer-{}'.format(layer_id))
    if args.verbose > 0:
        print('Report dir: {}'.format(report_dir))
    
    layer_problem_file = os.path.join(benchmark_dir, '{}_workload/{}.yaml'.format(workload, layer))
    with open(layer_problem_file, 'r') as fd:
        if args.verbose >= 2:
            print('Loading layer problem file and extend it: {}'.format(layer_problem_file))
        layer_problem = yaml.load(fd, Loader=yaml.SafeLoader)
        if args.verbose >= 2:
            print('\t', layer_problem)
        # FIXME:避免problem文件的硬编码，另外，此处定义的格式已经outdated
        problem = {'problem': {
            'shape': {'name': 'CNN-Layer', 'dimensions': ['H', 'C', 'K', 'R', 'S', 'N', 'P', 'Q'],
                      'coefficients': [{'name': 'Wstride', 'default': 1},
                                       {'name': 'Hstride', 'default': 1},
                                       {'name': 'Wdilation', 'default': 1},
                                       {'name': 'Hdilation', 'default': 1}],
                      },
            'instance': {'C': 256, 'K': 512, 'R': 3, 'S': 3, 'P': 56, 'Q': 56, 'H': 1, 'N': 16,
                         'Wstride': 1, 'Hstride': 1, 'Wdilation': 1, 'Hdilation': 1
                         }}}
        if 'type' in layer_problem['problem'].keys() and layer_problem['problem']['type'] == 'T2D':
            problem['problem']['shape']['data-spaces'] = [
                              {'name': 'Weights',
                               'projection': [[['H']], [['C']], [['K']], [['R']], [['S']]]},
                              {'name': 'Outputs', 'projection': [[['N']], [['H']], [['K']],
                                                                [['R', 'Wdilation'],
                                                                 ['P', 'Wstride']],
                                                                [['S', 'Hdilation'],
                                                                 ['Q', 'Hstride']]],
                               'read-write': True},
                              {'name': 'Inputs', 'projection': [[['N']], [['H']], [['C']], [['Q']], [['P']]]}]
            problem['problem']['instance']['type'] = 'T2D'
        else:
            problem['problem']['shape']['data-spaces'] = [
                              {'name': 'Weights',
                               'projection': [[['H']], [['C']], [['K']], [['R']], [['S']]]},
                              {'name': 'Inputs', 'projection': [[['N']], [['H']], [['C']],
                                                                [['R', 'Wdilation'],
                                                                 ['P', 'Wstride']],
                                                                [['S', 'Hdilation'],
                                                                 ['Q', 'Hstride']]]},
                              {'name': 'Outputs',
                               'projection': [[['N']], [['H']], [['K']], [['Q']], [['P']]],
                               'read-write': True}]
            problem['problem']['instance']['type'] = 'C2D'
        if 'H' in layer_problem['problem'].keys():
            problem['problem']['instance']['H'] = layer_problem['problem']['H']
        else:
            problem['problem']['instance']['H'] = 1
        if 'type' in layer_problem['problem'].keys() and layer_problem['problem']['type'] == 'BMM':
            problem['problem']['instance']['N'] = layer_problem['problem']['N']
            problem['problem']['instance']['H'] = layer_problem['problem']['H'] * batch_size
        else:
            problem['problem']['instance']['N'] = layer_problem['problem']['N'] * batch_size
        problem['problem']['instance']['K'] = layer_problem['problem']['K']
        problem['problem']['instance']['C'] = layer_problem['problem']['C']
        problem['problem']['instance']['P'] = layer_problem['problem']['P']
        problem['problem']['instance']['Q'] = layer_problem['problem']['Q']
        problem['problem']['instance']['R'] = layer_problem['problem']['R']
        problem['problem']['instance']['S'] = layer_problem['problem']['S']
        problem['problem']['instance']['Wstride'] = layer_problem['problem']['Wstride']
        problem['problem']['instance']['Hstride'] = layer_problem['problem']['Hstride']
        problem['problem']['instance']['Wdilation'] = layer_problem['problem']['Wdilation']
        problem['problem']['instance']['Hdilation'] = layer_problem['problem']['Hdilation']
    
    
    problem_file = os.path.join(accelerator_dir, accelerator, 'problem.yaml')
    with open(problem_file, 'w') as fd:
        if args.verbose >= 2:
            print('Dumping extended problem file: {}'.format(problem_file))
        yaml.dump(problem, fd)
        
    tuner = Tuner(problem['problem']['instance'], accelerator, report_dir, args.optim_obj, verbose=args.verbose)
    chkpt = tuner.run(args.epochs)
    os.makedirs(report_dir, exist_ok=True)
    with open(os.path.join(report_dir, 'env_chkpt.plt'), 'wb') as fd:
        pickle.dump(chkpt, fd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--optim_obj', type=str, default="latency", help='optimization objective')
    parser.add_argument('--epochs', type=int, default=10, help='number of generations/epochs')
    parser.add_argument('--verbose', type=int, default=2, help='logging level')
    parser.add_argument('--report_dir', type=str, default='./report', help='The report directory')

    parser.add_argument('--accelerator', type=str, default='arch', help='accelerator accelerator')
    parser.add_argument('--workload', type=str, default=None)
    parser.add_argument('--layer_id', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    args = parser.parse_args()

    ##############################
    # CUDA_VISIBLE_DEVICES=0 python main.py --optim_obj latency --epochs 10 --accelerator Simba --workload resnet50 --layer_id 43 --batch_size 16
    args.optim_obj = 'latency'
    args.epochs = 10
    args.accelerator = 'Simba'
    args.workload = 'resnet50'
    args.layer_id = 43
    args.batch_size = 16
    
    ##############################
    
    main()
