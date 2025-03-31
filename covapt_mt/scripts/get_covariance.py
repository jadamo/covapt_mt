from .make_gaussian_covariance import make_gaussian_covariance
from .make_window_function import make_window_function

def get_covariance(yaml_file):

    make_window_function(yaml_file)
    make_gaussian_covariance(yaml_file)


if __name__ == "__main__":
    """
    Example usage:
        python3 -m covapt_mt.scripts.get_covariance ./config/get_covariance.yaml
    """

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config_file", type=str, default=None,
        help="path to config file."
    )

    command_line_args = parser.parse_args()

    yaml_file = command_line_args.config_file

    get_covariance(yaml_file)