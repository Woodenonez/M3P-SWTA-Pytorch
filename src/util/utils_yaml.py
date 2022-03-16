import os
import yaml

from pathlib import Path

'''
This is used to load and dump parameters in the form of YAML
'''

file_name = 'sdd_ewta_15_test.yml'
sl_path = os.path.join(Path(__file__).resolve().parents[2], 'Config/', file_name)

general_param = {'with_T'   : False, 
                 'past'     : 4, 
                 'cpi'      : 2, 
                 'dim_out'  : 2, 
                #  'num_hypos': 20, 
                #  'num_gaus' : 5,
                 'num_components': 20,
                 'fc_input' : 23040, # 4608, 23040
                 'device'   : 'cuda',
                 }
general_param['input_channel'] = (general_param['past']+1) * general_param['cpi'] + general_param['with_T']

training_param = {'epoch'            : 20, 
                  'validation_prop'  : 0.2, 
                  'batch_size'       : 30, 
                  'early_stopping'   : 0,
                  }

path_param = {'model_path': 'Model/ewta_20m_20_sdd',
              'data_root':  'Data/',
              'data_name':  'SDD_3FPS_Test',
              'label_csv':  'all_data.csv',
              'zip_path':   'Data/SDD_3FPS_Test.zip', # valid for zip
              }

def to_yaml(data, save_path):
    with open(save_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
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

if __name__ == '__main__':

    param = {**general_param, **training_param, **path_param}
    to_yaml(param, sl_path)
    test_dict = from_yaml(sl_path) # ensure the yaml file is saved