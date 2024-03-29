"""
Perform miscellaneous experiments here.
"""
from pathlib import Path

from constants import APP_ROOT_DIR
from src.utils.run_config import load_yaml_config
from src.utils.nasbench201_configspace import get_arch_performance

# ["Identity", "Zero", "ReLUConvBN3x3", "ReLUConvBN1x1", "AvgPool3x3"]

if __name__ == "__main__":
    config = load_yaml_config(path_to_yaml=Path(APP_ROOT_DIR / 'configs' / 'run_config.yaml'))
    arch = ["ReLUConvBN3x3", "ReLUConvBN3x3", "Identity", "AvgPool3x3", "Zero", "ReLUConvBN3x3"]
    train_loss, val_loss, test_loss, train_regret, val_regret, test_regret, train_time = get_arch_performance(arch,
                                                                                                              config)
    print(f'Architecture: \n{arch}\n')
    print(f'train_loss: {train_loss}\n'
          f'val_loss: {val_loss}\n'
          f'test_loss: {test_loss}\n'
          f'train_regret: {train_regret}\n'
          f'val_regret: {val_regret}\n'
          f'test_regret: {test_regret}\n'
          f'train_time: {train_time}\n')

