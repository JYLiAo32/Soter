import pytimeloop.timeloopfe.v4 as tl
import os
import inspect

THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


path_arch = "/home/ljy/workspace/mapping/Soter/test/v4/arch.yaml"
path_prob = "/home/ljy/workspace/mapping/Soter/test/v4/problem.yaml"
path_map = "/home/ljy/workspace/mapping/Soter/test/v4/map.yaml"
output_path = "/home/ljy/workspace/mapping/Soter/test/v4/output"


def run_timeloop_model(
    arch_path: str,
    map_path: str,
    prob_path: str,
    out_dir: str,
):
    spec = tl.Specification.from_yaml_files(arch_path, map_path, prob_path)
    tl.call_model(spec, output_dir=out_dir)


if __name__ == "__main__":
    THIS_SCRIPT_DIR = os.path.dirname(
        os.path.abspath(inspect.getfile(inspect.currentframe()))
    )
    run_timeloop_model(path_arch, path_map, path_prob, output_path)
