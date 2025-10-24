import pytimeloop.timeloopfe.v4 as tl
import os
import inspect

THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))


path_arch = "/home/ljy/workspace/mapping/Soter/test/v4_soter/simba_arch.yaml"
path_prob = "/home/ljy/workspace/mapping/Soter/test/v4_soter/problem.yaml"
path_map = "/home/ljy/workspace/mapping/Soter/test/v4_soter/map.yaml"
path_component = "/home/ljy/workspace/mapping/Soter/test/v4_soter/components/*.yaml"
output_path = "/home/ljy/workspace/mapping/Soter/test/v4_soter/output"
path_mapper = "/home/ljy/workspace/mapping/Soter/test/v4_soter/mapper.yaml"


def run_01():
    spec = tl.Specification.from_yaml_files(
            path_arch,
            path_component,
            path_prob,
            path_map,
        )
    tl.call_model(spec, output_dir=os.path.join(output_path, "model"))

def run_02():
    spec = tl.Specification.from_yaml_files(
            path_arch,
            path_component,
            path_prob,
            path_mapper,
        )
    tl.call_mapper(spec, output_dir=os.path.join(output_path, "mapping"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run timeloop exercises")
    parser.add_argument(
        "exercise",
        type=str,
        help="Exercise to run. 'All' to run all exercises",
        default="",
        nargs="*",
    )
    parser.add_argument(
        "--generate-ref-outputs", action="store_true", help="Generate reference output"
    )
    parser.add_argument(
        "--clear-outputs", action="store_true", help="Clear output directories"
    )
    args = parser.parse_args()
    
    exercise_list = [
        "01",
        "02",
    ]

    if args.clear_outputs:
        prefixes = list(set([e.split("_")[0] for e in exercise_list]))

        for p in prefixes:
            path = os.path.abspath(os.path.join(THIS_SCRIPT_DIR, p + "*"))
            os.system(f"cd {path} ; rm -rf output")
            os.system(f"cd {path} ; rm -rf ref-output")
        exit()

    def run(target):
        func = globals()[f"run_{target}"]
        # Print out the function contents so users can see what the function does
        print("\n\n" + "=" * 80)
        print(f"Calling exercise {target}. Code is:")
        print(inspect.getsource(func))
        if args.generate_ref_outputs:
            func("ref-output")
        else:
            func()

    for e in args.exercise:
        if e.lower() == "all":
            for e in exercise_list:
                run(e)
        else:
            run(e)
