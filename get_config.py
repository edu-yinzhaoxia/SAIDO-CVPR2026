import argparse
import yaml
def get_config():    
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--yaml")
    args = argparser.parse_args()
    yaml_file_path=args.yaml
    cfg = yaml.load(open(yaml_file_path, 'r'), Loader=yaml.Loader)
    for key in cfg.keys():
        sub_dict=cfg[key]
        for keyy in sub_dict.keys():
            vars(args)[keyy]=sub_dict[keyy]
    return args