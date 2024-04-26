import os
import sys
import random
from time import time

import pandas as pd
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from src.data_loader.loader_bprmf import DataLoaderBPRMF
from src.model.BPRMF import BPRMF
from src.parser.parser_bprmf import *
from src.utils.log_helper import *
from src.utils.metrics import *
from src.utils.model_helper import *
from src.utils.extract_course import extract_course

class BPRMFPipeline:
    # def __init__(self, args):
    #     self.data_loader = DataLoaderBPRMF(args, logging)

    
    def __call__(self, args):
        # GPU / CPU
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # load data
        data = DataLoaderBPRMF(args, logging)
        test_user_dict = data.test_user_dict
        user_ids = list(test_user_dict.keys())
        print(len(user_ids))
        
        model = BPRMF(args, data.n_users, data.n_items)
        model = load_model(model, args.pretrain_model_path)
        model.to(device)
        print("model loaded", model)
        
        # predict
        Ks = eval(args.Ks)
        k_min = min(Ks)
        k_max = max(Ks)
        
        cf_scores, metrics_dict = self.evaluate(model, data, Ks, device)
        
        print('CF Evaluation: Precision [{:.4f}, {:.4f}], Recall [{:.4f}, {:.4f}], NDCG [{:.4f}, {:.4f}]'.format(
        metrics_dict[k_min]['precision'], metrics_dict[k_max]['precision'], metrics_dict[k_min]['recall'], metrics_dict[k_max]['recall'], metrics_dict[k_min]['ndcg'], metrics_dict[k_max]['ndcg']))
        
        # Kiểm tra user có đăng kí khóa học chưa, nếu rồi thì gán khóa học đó là -999 (tránh khuyến nghị các khóa học đã học rồi)
        user_course_dict = np.load("src\\datasets\\user_enroll_course.npy", allow_pickle=True).item()
        for i in range(0, len(user_ids)):
            for course_ids in user_course_dict[user_ids[i]]:
                cf_scores[i, course_ids] = -999
        
        top_5_values_per_row = []
        top_5_indices_per_row = []
        
        for row in cf_scores:
            indices_descending = np.argsort(-row)
            
            top_5_values = row[indices_descending[:5]]
            top_5_indices = np.array(indices_descending[:5])
            
            top_5_values_per_row.append(top_5_values)
            top_5_indices_per_row.append(top_5_indices)
        
        return top_5_indices_per_row, user_ids
    
    def evaluate(self, model, dataloader, Ks, device):
        test_batch_size = dataloader.test_batch_size
        train_user_dict = dataloader.train_user_dict
        test_user_dict = dataloader.test_user_dict
    
        model.eval()
    
        user_ids = list(test_user_dict.keys())
        user_ids_batches = [user_ids[i: i + test_batch_size] for i in range(0, len(user_ids), test_batch_size)]
        user_ids_batches = [torch.LongTensor(d) for d in user_ids_batches]
    
        n_items = dataloader.n_items
        item_ids = torch.arange(n_items, dtype=torch.long).to(device)
    
        cf_scores = []
        metric_names = ['precision', 'recall', 'ndcg']
        metrics_dict = {k: {m: [] for m in metric_names} for k in Ks}
    
        with tqdm(total=len(user_ids_batches), desc='Evaluating Iteration') as pbar:
            for batch_user_ids in user_ids_batches:
                batch_user_ids = batch_user_ids.to(device)
    
                with torch.no_grad():
                    batch_scores = model(batch_user_ids, item_ids, is_train=False)       # (n_batch_users, n_items)
    
                batch_scores = batch_scores.cpu()
                batch_metrics = calc_metrics_at_k(batch_scores, train_user_dict, test_user_dict, batch_user_ids.cpu().numpy(), item_ids.cpu().numpy(), Ks)
    
                cf_scores.append(batch_scores.numpy())
                for k in Ks:
                    for m in metric_names:
                        metrics_dict[k][m].append(batch_metrics[k][m])
                pbar.update(1)
    
        cf_scores = np.concatenate(cf_scores, axis=0)
        for k in Ks:
            for m in metric_names:
                metrics_dict[k][m] = np.concatenate(metrics_dict[k][m]).mean()
        return cf_scores, metrics_dict

if __name__ == '__main__':
    args = parse_bprmf_args()
    
    args.data_dir = "src\\datasets"
    args.data_name = "mooccube"
    args.use_pretrain = 2
    args.Ks = '[1, 5, 10]'
    args.pretrain_model_path = 'src\\pretrained_model\\model_BPRMF.pth'
    
    print(args)
    
    pipeline = BPRMFPipeline()
    # print(pipeline(args))
    print(extract_course(pipeline(args)))
        
        