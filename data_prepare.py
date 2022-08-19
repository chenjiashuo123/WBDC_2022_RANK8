import json
import random
import numpy as np
import os
import zipfile
import numpy as np
from sklearn.model_selection import StratifiedKFold
from utils.category_id_map import category_id_to_lv2id, category_id_to_lv1id
from config import parse_args


def data_split_skf_fold(train_annotation, save_path, skf_fold=10):
    with open(train_annotation, 'r', encoding='utf8') as file:
        anns = json.load(file)
    data_label = []
    for idx in range(len(anns)):
        data_label.append(category_id_to_lv2id(anns[idx]['category_id']))
    print(len(data_label))
    sfolder = StratifiedKFold(n_splits=skf_fold,random_state=802,shuffle=True)
    fold = 0
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for train, test in sfolder.split(data_label,data_label):
        validation_path = save_path + f'/validation_idx_{fold}.npy'
        train_path = save_path + f'/train_idx_{fold}.npy'
        np.save(validation_path, test)
        np.save(train_path, train)
        print('Train: %s | test: %s' % (len(train), len(test)))
        fold = fold + 1
def gen_train_json(annotation, npy_name, json_save_name):
    with open(annotation, 'r', encoding='utf8') as file:
        anns = json.load(file)
    data_list = []
    idx = np.load(npy_name).tolist()
    for id in idx:
        data_list.append(anns[id])
    with open(json_save_name,'w',encoding = 'utf-8') as f:
        #print(len(data_list))
        data = json.dumps(data_list)
        f.write(data)

if __name__ == '__main__':
    args = parse_args()
    data_split_skf_fold(args.train_annotation, args.fold_save_path, args.skf_fold)
    
    npy_name = 'fold_data/five/train_idx_1.npy'
    json_save_name = 'fold_data/labeled-train.json'
    gen_train_json(args.train_annotation, npy_name, json_save_name)
    
    npy_name = 'fold_data/five/validation_idx_1.npy'
    json_save_name = 'fold_data/labeled-validation.json'
    gen_train_json(args.train_annotation, npy_name, json_save_name)

