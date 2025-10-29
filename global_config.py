from datetime import datetime



class GlobalConfig:
    PROBLEM_FILE = "problem.yaml"
    ARCHITECTURE_FILE = "arch.yaml"
    MAPPING_FILE = "mapping.yaml"
    MAPPER_FILE = "mapper.yaml"
    COMPONENT_PATH = "components/*.yaml"

    DUMP_ENV_CKPT_PATH = "env_chkpt.pkl"
    TIMELOOP_CONFIG_PATH = './SpatialAccelerators_v4'
    TIMELOOP_OUTPUT_PATH = f'./tmp/out_config_{datetime.now().strftime("%H:%M:%S")}'
    def __init__(self, args):
        self.args = args
