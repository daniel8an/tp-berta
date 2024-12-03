import os
import json
import numpy as np


def main():
    directory_path = "/home/danieleitan/tp-berta/finetune_outputs/binclass/TPBerta-default_epoch_10"
    datas = {}
    for dir_name in os.listdir(directory_path):
        with open(os.path.join(directory_path, dir_name, "finish.json"), "r") as file:
            data = json.load(file)
            datas[dir_name] = data
    print('bla')


if __name__ == "__main__":
    main()
