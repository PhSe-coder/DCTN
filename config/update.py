import yaml

import os.path as osp
from argparse import ArgumentParser

parser = ArgumentParser(description="Update config files")

parser.add_argument("--source", type=str, default="service")
parser.add_argument("--target", type=str, default="restaurant")
parser.add_argument("--dim", type=int, default=300)

args = parser.parse_args()
# update pretrain config yaml
with open("config/pretrain.yaml") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config["data"]["init_args"]["target"] = args.target
source_train_file = config["data"]["init_args"]["source_train_file"]
target_train_file = config["data"]["init_args"]["source_train_file"]
source = osp.basename(source_train_file).split('.')[0]
target = osp.basename(target_train_file).split('.')[0]
config["model"]["init_args"]["h_dim"] = args.dim
max_epochs = config["trainer"]["max_epochs"]
config["data"]["init_args"]["source_train_file"] = source_train_file.replace(source, args.source)
config["data"]["init_args"]["target_train_file"] = target_train_file.replace(target, args.target)
config["trainer"]["callbacks"][0]["init_args"][
    "filename"] = f"pretrained-fdgr-{args.source}-{args.target}-dim={args.dim}-max-epoch={max_epochs}"
with open("config/pretrain.yaml", "w") as f:
    yaml.dump(config, f)

# update train config yaml
with open("config/train.yaml") as f:
    train_config = yaml.load(f, Loader=yaml.FullLoader)
validation_file = train_config["data"]["init_args"].get("validation_file")
source = osp.basename(validation_file).split('.')[0]
train_config["data"]["init_args"]["validation_file"] = validation_file.replace(source, args.source)
train_config["data"]["init_args"]["target"] = args.target
train_config["data"]["init_args"]["source_train_file"] = source_train_file.replace(source, args.source)
train_config["data"]["init_args"]["target_train_file"] = target_train_file.replace(target, args.target)
dirpath = config["trainer"]["callbacks"][0]["init_args"]["dirpath"] 
filename = config["trainer"]["callbacks"][0]["init_args"]["filename"]
train_config["model"]["init_args"]["model"]["init_args"]["pretrained_path"] = osp.join(dirpath, filename+"-v1.ckpt")
with open("config/train.yaml", "w") as f:
    yaml.dump(train_config, f)

# update test config yaml
with open("config/test.yaml") as f:
    test_config = yaml.load(f, Loader=yaml.FullLoader)
test_config["model"]["init_args"]["model"]["init_args"]["pretrained_path"] = osp.join(dirpath, filename+"-v1.ckpt")
test_file = test_config["data"]["init_args"].get("test_file")
source = osp.basename(test_file).split('.')[0]
test_config["data"]["init_args"]["test_file"] = test_file.replace(source, args.target)
with open("config/test.yaml", "w") as f:
    yaml.dump(test_config, f)