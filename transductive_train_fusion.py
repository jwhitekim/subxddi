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
import custom_loss
from data_preprocessing_fusion import DrugDataset, DrugDataLoader
import warnings
warnings.filterwarnings('ignore',category=UserWarning)

######################### Parameters ######################
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
parser.add_argument('--n_atom_feats', type=int, default=55, help='num of input features')
parser.add_argument('--n_atom_hid', type=int, default=128, help='num of hidden features')
parser.add_argument('--rel_total', type=int, default=86, help='num of interaction types')
parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--n_epochs', type=int, default=200, help='num of epochs')
parser.add_argument('--kge_dim', type=int, default=128, help='dimension of interaction matrix')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')


parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--neg_samples', type=int, default=1)
parser.add_argument('--data_size_ratio', type=float, default=1)
parser.add_argument('--use_cuda', type=bool, default=True, choices=[0, 1])
parser.add_argument('--pkl_name', type=str, default='transductive_drugbank.pkl')
parser.add_argument('--fusion_mode', type=str, default='none',
                    choices=['none', 'concat', 'mlp', 'cross_attention'])
parser.add_argument('--sumgnn_dir', type=str, default=models.DEFAULT_SUMGNN_DIR)
parser.add_argument('--sumgnn_dim', type=int, default=384)
parser.add_argument('--sumgnn_max_paths', type=int, default=16)
parser.add_argument('--cross_attention_heads', type=int, default=4)
parser.add_argument('--train_scope', type=str, default='all',
                    choices=['all', 'decoder_only'])
parser.add_argument('--init_checkpoint', type=str, default=None,
                    help='Optional full-model/state_dict checkpoint to load before freezing train_scope.')
parser.add_argument('--output_dir', type=str, default=None)
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--target_train_loss', type=float, default=None)
parser.add_argument('--target_train_acc', type=float, default=None)
parser.add_argument('--early_stop_on_target', type=bool, default=False, choices=[0, 1])

args = parser.parse_args()


def parse_scalar(value):
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [parse_scalar(item) for item in inner.split(",")]
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("none", "null"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value.strip("'\"")


def load_simple_yaml(path):
    config = {}
    with open(path) as f:
        for raw_line in f:
            line = raw_line.split("#", 1)[0].strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = parse_scalar(value)
    return config


if args.config:
    config = load_simple_yaml(args.config)
    provided = {
        item.lstrip("-").replace("-", "_")
        for item in sys.argv[1:]
        if item.startswith("--")
    }
    for key, value in config.items():
        if hasattr(args, key) and key not in provided:
            setattr(args, key, value)

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
if args.output_dir is None:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args.output_dir = os.path.join("outputs", "fusion", f"{args.fusion_mode}_{stamp}")
os.makedirs(args.output_dir, exist_ok=True)
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

train_data_loader = DrugDataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=args.num_workers)
val_data_loader = DrugDataLoader(val_data, batch_size=batch_size *3,num_workers=args.num_workers)
test_data_loader = DrugDataLoader(test_data, batch_size=batch_size *3,num_workers=args.num_workers)


def move_tri_to_device(tri, device):
        return tuple(item.to(device=device) if hasattr(item, 'to') else item for item in tri)


def get_scores(output):
        return output['scores'] if isinstance(output, dict) else output


def do_compute(batch, device, model):
        '''
            *batch: (pos_tri, neg_tri)
            *pos/neg_tri: (batch_h, batch_t, batch_r)
        '''
        probas_pred, ground_truth = [], []
        pos_tri, neg_tri = batch
        
        pos_tri = move_tri_to_device(pos_tri, device)
        p_score = get_scores(model(pos_tri))
        probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
        ground_truth.append(np.ones(len(p_score)))

        neg_tri = move_tri_to_device(neg_tri, device)
        n_score = get_scores(model(neg_tri))
        probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
        ground_truth.append(np.zeros(len(n_score)))

        probas_pred = np.concatenate(probas_pred)
        ground_truth = np.concatenate(ground_truth)

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


def configure_train_scope(model, train_scope):
    if train_scope == 'all':
        return

    for param in model.parameters():
        param.requires_grad = False

    trainable_modules = [model.decoder_fusion, model.KGE]
    if model.fusion_mode == 'none':
        trainable_modules.append(model.co_attention)

    for module in trainable_modules:
        for param in module.parameters():
            param.requires_grad = True


def count_trainable_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def apply_runtime_paths(model, sumgnn_dir):
    if hasattr(model, 'sumgnn_dir'):
        model.sumgnn_dir = sumgnn_dir
    if hasattr(model, 'sumgnn_encoder') and model.sumgnn_encoder is not None:
        model.sumgnn_encoder.sumgnn_dir = sumgnn_dir
        if hasattr(model.sumgnn_encoder, '_loaded'):
            model.sumgnn_encoder._loaded = False


def load_initial_checkpoint(path, model, sumgnn_dir):
    checkpoint = torch.load(path, map_location='cpu')

    if isinstance(checkpoint, torch.nn.Module):
        if hasattr(checkpoint, 'fusion_mode') and checkpoint.fusion_mode != model.fusion_mode:
            raise ValueError(
                f"Checkpoint fusion_mode={checkpoint.fusion_mode} does not match "
                f"--fusion_mode={model.fusion_mode}"
            )
        apply_runtime_paths(checkpoint, sumgnn_dir)
        print(f"Loaded initial full model checkpoint: {path}")
        return checkpoint

    if not isinstance(checkpoint, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)}")

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded initial model_state_dict checkpoint: {path}")
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
        print(f"Loaded initial decoder-only checkpoint: {path}")
        return model

    model.load_state_dict(checkpoint)
    print(f"Loaded initial raw state_dict checkpoint: {path}")
    return model


def decoder_state_dict(model):
    state = {
        'fusion_mode': model.fusion_mode,
        'decoder_fusion': model.decoder_fusion.state_dict(),
        'KGE': model.KGE.state_dict(),
    }
    if model.fusion_mode == 'none':
        state['co_attention'] = model.co_attention.state_dict()
    else:
        state['co_attention'] = 'skipped_for_fusion_mode'
    return state


def save_decoder_weights(model, output_dir, suffix):
    path = os.path.join(output_dir, f'decoder_{suffix}.pt')
    torch.save(decoder_state_dict(model), path)
    return path


def save_model_state_dict(model, output_dir, suffix):
    path = os.path.join(output_dir, f'model_{suffix}_state_dict.pkl')
    torch.save(model.state_dict(), path)
    return path


def write_metrics_csv(history, path):
    columns = [
        'epoch', 'train_loss', 'val_loss', 'train_acc', 'val_acc',
        'train_roc', 'val_roc', 'train_precision', 'val_precision',
    ]
    with open(path, 'w') as f:
        f.write(','.join(columns) + '\n')
        for row in history:
            f.write(','.join(str(row[col]) for col in columns) + '\n')


def svg_line_plot(history, series, title, path):
    width, height = 760, 420
    margin = 54
    plot_w = width - 2 * margin
    plot_h = height - 2 * margin
    colors = ['#2563eb', '#dc2626', '#059669', '#7c3aed']
    values = []
    for name in series:
        values.extend([row[name] for row in history])
    if not values:
        return
    y_min, y_max = min(values), max(values)
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0

    def x_pos(index):
        if len(history) == 1:
            return margin + plot_w / 2
        return margin + index * plot_w / (len(history) - 1)

    def y_pos(value):
        return margin + (y_max - value) * plot_h / (y_max - y_min)

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{margin}" y="30" font-family="Arial" font-size="20" fill="#111827">{title}</text>',
        f'<line x1="{margin}" y1="{height-margin}" x2="{width-margin}" y2="{height-margin}" stroke="#6b7280"/>',
        f'<line x1="{margin}" y1="{margin}" x2="{margin}" y2="{height-margin}" stroke="#6b7280"/>',
    ]

    for idx, name in enumerate(series):
        points = ' '.join(
            f'{x_pos(i):.2f},{y_pos(row[name]):.2f}'
            for i, row in enumerate(history)
        )
        color = colors[idx % len(colors)]
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}"/>')
        lines.append(
            f'<text x="{margin + idx * 150}" y="{height - 14}" '
            f'font-family="Arial" font-size="13" fill="{color}">{name}</text>'
        )

    lines.append(
        f'<text x="{margin}" y="{height-margin+34}" font-family="Arial" '
        f'font-size="12" fill="#374151">epoch</text>'
    )
    lines.append(
        f'<text x="{margin}" y="{margin-10}" font-family="Arial" '
        f'font-size="12" fill="#374151">{y_max:.4f}</text>'
    )
    lines.append(
        f'<text x="{margin}" y="{height-margin-6}" font-family="Arial" '
        f'font-size="12" fill="#374151">{y_min:.4f}</text>'
    )
    lines.append('</svg>')

    with open(path, 'w') as f:
        f.write('\n'.join(lines))


def save_history_outputs(history, output_dir):
    write_metrics_csv(history, os.path.join(output_dir, 'metrics.csv'))
    svg_line_plot(
        history,
        ['train_loss', 'val_loss'],
        'Loss by Epoch',
        os.path.join(output_dir, 'loss_plot.svg'),
    )
    svg_line_plot(
        history,
        ['train_acc', 'val_acc', 'train_roc', 'val_roc'],
        'Accuracy and ROC by Epoch',
        os.path.join(output_dir, 'acc_roc_plot.svg'),
    )


def train(model, train_data_loader, val_data_loader, loss_fn,  optimizer, n_epochs, device, scheduler=None, output_dir=None):
    max_acc = -1
    history = []
    print('Starting training at', datetime.today())
    for i in range(1, n_epochs+1):
        start = time.time()
        total_train_batches = len(train_data_loader)
        print(f'[Epoch {i}/{n_epochs}] start - {total_train_batches} train batches', flush=True)
        train_loss = 0
        train_loss_pos = 0
        train_loss_neg = 0
        val_loss = 0
        val_loss_pos = 0
        val_loss_neg = 0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []
       
        for batch_idx, batch in enumerate(train_data_loader, start=1):
         
            model.train()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)
        
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
           
            train_loss += loss.item() * len(p_score)
            if batch_idx == 1 or batch_idx % 100 == 0 or batch_idx == total_train_batches:
                print(
                    f'[Epoch {i}/{n_epochs}] train batch {batch_idx}/{total_train_batches} '
                    f'loss={loss.item():.4f}',
                    flush=True,
                )
        train_loss /= len(train_data)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_auc_roc, train_f1, train_precision,train_recall,train_int_ap, train_ap = do_compute_metrics(train_probas_pred, train_ground_truth)

            total_val_batches = len(val_data_loader)
            print(f'[Epoch {i}/{n_epochs}] validation start - {total_val_batches} val batches', flush=True)
            for val_batch_idx, batch in enumerate(val_data_loader, start=1):
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                val_probas_pred.append(probas_pred)
                val_ground_truth.append(ground_truth)
                loss, loss_p, loss_n = loss_fn(p_score, n_score)
                val_loss += loss.item() * len(p_score)            
                if val_batch_idx == 1 or val_batch_idx % 50 == 0 or val_batch_idx == total_val_batches:
                    print(
                        f'[Epoch {i}/{n_epochs}] val batch {val_batch_idx}/{total_val_batches} '
                        f'loss={loss.item():.4f}',
                        flush=True,
                    )

            val_loss /= len(val_data)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_acc, val_auc_roc, val_f1, val_precision,val_recall,val_int_ap, val_ap = do_compute_metrics(val_probas_pred, val_ground_truth)
            if val_acc>max_acc:
                max_acc = val_acc
                torch.save(model, pkl_name)
                if output_dir:
                    save_decoder_weights(model, output_dir, 'best')
                    save_model_state_dict(model, output_dir, 'best')
               
        if scheduler:
            # print('scheduling')
            scheduler.step()


        history.append({
            'epoch': i,
            'train_loss': round(train_loss, 6),
            'val_loss': round(val_loss, 6),
            'train_acc': round(train_acc, 6),
            'val_acc': round(val_acc, 6),
            'train_roc': round(train_auc_roc, 6),
            'val_roc': round(val_auc_roc, 6),
            'train_precision': round(train_precision, 6),
            'val_precision': round(val_precision, 6),
        })
        if output_dir:
            save_history_outputs(history, output_dir)
            save_decoder_weights(model, output_dir, 'last')

        print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f},'
        f' train_acc: {train_acc:.4f}, val_acc:{val_acc:.4f}', flush=True)
        print(f'\t\ttrain_roc: {train_auc_roc:.4f}, val_roc: {val_auc_roc:.4f}, train_precision: {train_precision:.4f}, val_precision: {val_precision:.4f}', flush=True)

        hit_loss = args.target_train_loss is None or train_loss <= args.target_train_loss
        hit_acc = args.target_train_acc is None or train_acc >= args.target_train_acc
        if args.early_stop_on_target and hit_loss and hit_acc:
            print(
                f'[Target reached] epoch={i}, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}',
                flush=True,
            )
            if output_dir:
                save_decoder_weights(model, output_dir, 'target')
                with open(os.path.join(output_dir, 'target_reached.txt'), 'w') as f:
                    f.write(
                        f'epoch={i}\ntrain_loss={train_loss:.6f}\ntrain_acc={train_acc:.6f}\n'
                    )
            break
    return history

def test(test_data_loader,model):
    test_probas_pred = []
    test_ground_truth = []
    with torch.no_grad():
        for batch in test_data_loader:
            model.eval()
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
if args.init_checkpoint:
    model = load_initial_checkpoint(args.init_checkpoint, model, args.sumgnn_dir)
loss = custom_loss.SigmoidLoss()
configure_train_scope(model, args.train_scope)
trainable_params = [param for param in model.parameters() if param.requires_grad]
print(f"Train scope: {args.train_scope}, trainable params: {count_trainable_params(model):,}")
optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 0.96 ** (epoch))
# print(model)
model.to(device=device)
# # if __name__ == '__main__':
train(model, train_data_loader, val_data_loader, loss, optimizer, n_epochs, device, scheduler, output_dir=args.output_dir)
save_decoder_weights(model, args.output_dir, 'final')
save_model_state_dict(model, args.output_dir, 'final')
test_model = torch.load(pkl_name)
test(test_data_loader,test_model)
