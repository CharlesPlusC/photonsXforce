from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
drive_path = str(PROJECT_ROOT / 'data') + '/'

def get_mesh_name(object_name):
    return drive_path + '3d/' + object_name + '.obj'

def get_force_path(object_name, design_name):
    return drive_path + 'force/' + object_name + '_' + design_name

def get_nn_weight_path(object_name, design_name):
    return drive_path + 'nn_weights/' + object_name + '_' + design_name + '/'