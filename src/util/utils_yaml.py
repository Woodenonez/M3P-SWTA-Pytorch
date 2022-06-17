import os
import yaml

from pathlib import Path

'''
This is used to load and dump parameters in the form of YAML
'''

file_name = 'sdd_15_ewta_test.yml'
sl_path = os.path.join(Path(__file__).resolve().parents[2], 'Config/', file_name)

general_param = {'with_T'   : False, 
                 'past_len'   : 4, 
                 'dim_out'  : 2, 
                #  'num_gaus' : 5,
                 'num_hypos': 20,
                 'fc_input' : 5760, # 5760, 23040
                 'dynamic_env': True,
                 'device'   : 'multi', # cpu, cuda, multi
                 }
general_param['input_channel'] = (general_param['past_len']+1)*2 + general_param['with_T']

training_param = {'epoch'            : 20, 
                  'validation_prop'  : 0.2, 
                  'batch_size'       : 30, 
                  'early_stopping'   : 0,
                  }

path_param = {'model_path': 'Model/sdd_15_ewta',
              'data_name':  'SDD_15_test',
              'label_csv':  'all_data.csv',
            #   'seg_name':   '4_0.png'
            #   'load_path':  'Model/ewta_20m_15_sdd', # valid for MDF
              }
path_param['data_path']  = os.path.join('Data/', path_param['data_name'])
path_param['label_path'] = os.path.join('Data/', path_param['data_name'], path_param['label_csv'])

def to_yaml(data, save_path, style=None):
    with open(save_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, default_style=style)
    print(f'Save to {save_path}.')
    return 0

def to_yaml_all(data_list, save_path, style=None):
    with open(save_path, 'w') as f:
        yaml.dump_all(data_list, f, explicit_start=True, default_flow_style=False, default_style=style)
    print(f'Save to {save_path}.')
    return 0

def from_yaml(load_path):
    with open(load_path, 'r') as stream:
        try:
            parsed_yaml = yaml.safe_load(stream)
            print(f'Load from {load_path}.')
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml

def from_yaml_all(load_path):
    with open(load_path, 'r') as stream:
        parsed_yaml_list = []
        try:
            for data in yaml.load_all(stream, Loader=yaml.FullLoader):
                parsed_yaml_list.append(data)
            print(f'Load from {load_path}.')
        except yaml.YAMLError as exc:
            print(exc)
    return parsed_yaml_list

