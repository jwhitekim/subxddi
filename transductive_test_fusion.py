from datetime import datetime
import time
import argparse
import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "dsn_encoder"))
import torch

from torch import optim
from sklearn import metrics
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
import models_fusion as models
from data_preprocessing_fusion import DrugDataset, DrugDataLoader
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--n_atom_feats', type=int, default=55, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=128, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=86, help='num of interaction types')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=200, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')


parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=int, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
parser.add_argument('--pkl_name', type=str, default='drugbank_test/transductive_drugbank.pkl')
parser.add_argument('--fusion_mode', type=str, default='none',
                    choices=['none', 'concat', 'mlp', 'cross_attention'])
parser.add_argument('--sumgnn_dir', type=str, default=models.DEFAULT_SUMGNN_DIR)
parser.add_argument('--sumgnn_dim', type=int, default=384)
parser.add_argument('--sumgnn_max_paths', type=int, default=16)
parser.add_argument('--cross_attention_heads', type=int, default=4)
parser.add_argument('--return_reasoning', action='store_true')

args = parser.parse_args()
n_atom_feats = args.n_atom_feats
n_atom_hid = args.n_atom_hid
rel_total = args.rel_total
lr = args.lr
n_epochs = args.n_epochs
kge_dim = args.kge_dim
batch_size = args.batch_size
pkl_name = args.pkl_name

weight_decay = args.weight_decay
neg_samples = args.neg_samples
data_size_ratio = args.data_size_ratio
device = 'cuda:0' if torch.cuda.is_available() and args.use_cuda else 'cpu'
print(args)
############################################################
###### Dataset
def split_train_valid(data, fold, val_ratio=0.2):
    data = np.array(data)
    cv_split = StratifiedShuffleSplit(n_splits=2, test_size=val_ratio, random_state=fold)
    train_index, val_index = next(iter(cv_split.split(X=data, y=data[:, 2])))
    train_tup = data[train_index]
    val_tup = data[val_index]
    train_tup = [(tup[0],tup[1],int(tup[2]))for tup in train_tup ]
    val_tup = [(tup[0],tup[1],int(tup[2]))for tup in val_tup ]

    return train_tup, val_tup

df_ddi_train = pd.read_csv('dataset/drugbank/fold0/train.csv')
df_ddi_test = pd.read_csv('dataset/drugbank/fold0/test.csv')

train_tup = [(h, t, r) for h, t, r in zip(df_ddi_train['d1'], df_ddi_train['d2'], df_ddi_train['type'])]
train_tup, val_tup = split_train_valid(train_tup,2, val_ratio=0.2)
test_tup = [(h, t, r) for h, t, r in zip(df_ddi_test['d1'], df_ddi_test['d2'], df_ddi_test['type'])]

train_data = DrugDataset(train_tup, ratio=data_size_ratio, neg_ent=neg_samples)
val_data = DrugDataset(val_tup, ratio=data_size_ratio, disjoint_split=False)
test_data = DrugDataset(test_tup, disjoint_split=False)


print(f"Training with {len(train_data)} samples, validating with {len(val_data)}, and testing with {len(test_data)}")

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=2)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size *3,num_workers=2)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size *3,num_workers=2)


def move_tri_to_device(tri, device):
        return tuple(item.to(device=device) if hasattr(item, 'to') else item for item in tri)


def model_forward(model, tri, return_reasoning=False):
        call_tri = tri if hasattr(model, 'fusion_mode') else tri[:4]
        if return_reasoning:
            try:
                return model(call_tri, return_reasoning=True)
            except TypeError:
                return model(call_tri)
        return model(call_tri)


def get_scores(output):
        return output['scores'] if isinstance(output, dict) else output


def get_reasoning(output):
        return output.get('reasoning_paths', []) if isinstance(output, dict) else []


def do_compute(batch, device, model, return_reasoning=False):
        '''
            *batch: (pos_tri, neg_tri)
            *pos/neg_tri: (batch_h, batch_t, batch_r)
        '''
        probas_pred, ground_truth = [], []
        reasoning = []
        pos_tri, neg_tri = batch
        
        pos_tri = move_tri_to_device(pos_tri, device)
        p_output = model_forward(model, pos_tri, return_reasoning=return_reasoning)
        p_score = get_scores(p_output)
        probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
        ground_truth.append(np.ones(len(p_score)))
        if return_reasoning:
            reasoning.extend(get_reasoning(p_output))

        neg_tri = move_tri_to_device(neg_tri, device)
        n_output = model_forward(model, neg_tri, return_reasoning=return_reasoning)
        n_score = get_scores(n_output)
        probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
        ground_truth.append(np.zeros(len(n_score)))
        if return_reasoning:
            reasoning.extend(get_reasoning(n_output))

        probas_pred = np.concatenate(probas_pred)
        ground_truth = np.concatenate(ground_truth)

        if return_reasoning:
            return p_score, n_score, probas_pred, ground_truth, reasoning
        return p_score, n_score, probas_pred, ground_truth


def do_compute_metrics(probas_pred, target):
    pred = (probas_pred >= 0.5).astype(int)
    acc = metrics.accuracy_score(target, pred)
    auroc = metrics.roc_auc_score(target, probas_pred)
    f1_score = metrics.f1_score(target, pred)
    precision = metrics.precision_score(target, pred)
    recall = metrics.recall_score(target, pred)
    p, r, t = metrics.precision_recall_curve(target, probas_pred)
    int_ap = metrics.auc(r, p)
    ap= metrics.average_precision_score(target, probas_pred)

    return acc, auroc, f1_score, precision, recall, int_ap, ap

def test(test_data_loader,model):
    test_probas_pred = []
    test_ground_truth = []
    reasoning_outputs = []
    with torch.no_grad():
        for batch in test_data_loader:
            model.eval()
            if args.return_reasoning:
                p_score, n_score, probas_pred, ground_truth, reasoning = do_compute(
                    batch, device, model, return_reasoning=True
                )
                reasoning_outputs.extend(reasoning)
            else:
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            test_probas_pred.append(probas_pred)
            test_ground_truth.append(ground_truth)
        test_probas_pred = np.concatenate(test_probas_pred)
        test_ground_truth = np.concatenate(test_ground_truth)
        test_acc, test_auc_roc, test_f1, test_precision,test_recall,test_int_ap, test_ap = do_compute_metrics(test_probas_pred, test_ground_truth)
    print('\n')
    print('============================== Test Result ==============================')
    print(f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f},test_f1: {test_f1:.4f},test_precision:{test_precision:.4f}')
    print(f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')
    if args.return_reasoning:
        print('=========================== Reasoning Sample ===========================')
        for item in reasoning_outputs[:3]:
            print(item)


def apply_runtime_paths(model, sumgnn_dir):
    if hasattr(model, 'sumgnn_dir'):
        model.sumgnn_dir = sumgnn_dir
    if hasattr(model, 'sumgnn_encoder') and model.sumgnn_encoder is not None:
        model.sumgnn_encoder.sumgnn_dir = sumgnn_dir
        if hasattr(model.sumgnn_encoder, '_loaded'):
            model.sumgnn_encoder._loaded = False


def load_test_model(path, model, sumgnn_dir):
    checkpoint = torch.load(path, map_location='cpu')

    if isinstance(checkpoint, torch.nn.Module):
        apply_runtime_paths(checkpoint, sumgnn_dir)
        print(f"Loaded full model checkpoint: {path}")
        return checkpoint

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model_state_dict checkpoint: {path}")
        return model

    decoder_keys = {'decoder_fusion', 'KGE'}
    if decoder_keys.issubset(checkpoint.keys()):
        ckpt_mode = checkpoint.get('fusion_mode')
        if ckpt_mode is not None and ckpt_mode != model.fusion_mode:
            raise ValueError(
                f"Checkpoint fusion_mode={ckpt_mode} does not match "
                f"--fusion_mode={model.fusion_mode}"
            )
        model.decoder_fusion.load_state_dict(checkpoint['decoder_fusion'])
        model.KGE.load_state_dict(checkpoint['KGE'])
        co_attention_state = checkpoint.get('co_attention')
        if isinstance(co_attention_state, dict):
            model.co_attention.load_state_dict(co_attention_state)
        print(f"Loaded decoder-only checkpoint: {path}")
        print("Warning: DSN encoder weights were not included in this checkpoint; "
              "the encoder remains newly initialized.")
        return model

    model.load_state_dict(checkpoint)
    print(f"Loaded raw state_dict checkpoint: {path}")
    return model


model = models.MVN_DDI(
    n_atom_feats,
    n_atom_hid,
    kge_dim,
    rel_total,
    heads_out_feat_params=[64,64,64,64],
    blocks_params=[2, 2, 2, 2],
    fusion_mode=args.fusion_mode,
    sumgnn_dir=args.sumgnn_dir,
    sumgnn_dim=args.sumgnn_dim,
    sumgnn_max_paths=args.sumgnn_max_paths,
    cross_attention_heads=args.cross_attention_heads,
)
# print(model)
# # if __name__ == '__main__':
test_model = load_test_model(pkl_name, model, args.sumgnn_dir)
test_model.to(device=device)
test(test_data_loader,test_model)
