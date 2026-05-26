import os
import sys
import argparse
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from seq_model import *
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from seq_dataset import *

import warnings
warnings.filterwarnings('ignore')

# --- Metric instrumentation (passive logging) ---
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
from shared.metric_logger import (
    HEMORRHAGE_CLASSES,
    GradNormAccumulator,
    MetricLogger,
    compute_full_metrics,
    flatten_for_csv,
    reset_system_metrics,
    system_metrics,
)

model_save_dir = os.path.join(final_output_path, 'version3_debug')
if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)

def bce_loss(input, target, OHEM_percent=None, class_num = None):
    if OHEM_percent is None:
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='mean')
        return loss
    else:
        loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')
        value, index= loss.topk(int(class_num * OHEM_percent), dim=1, largest=True, sorted=True)
        return value.mean()

def criterion(logit, labels):
    w = [2.0,1.0,1.0,1.0,1.0,1.0]
    loss = [bce_loss(logit[:, 0, :, i:i+1], labels[:,:, i:i+1])*w[i] for i in range(6)]
    loss = sum(loss) / sum(w)
    return loss

log = open(os.path.join(model_save_dir,'log.txt'),'a')

def _collect_val_predictions(model, val_loader):
    """Run validation and return numpy arrays for SM2 logits, SM1 logits, and labels.

    Returns
    -------
    sm2_probs : (N, 6)  -- sigmoid of `logit` (final SM2 output)
    sm1_probs : (N, 6)  -- sigmoid of `logit_help` (SM1 auxiliary output)
    gt        : (N, 6)  -- ground truth labels concatenated across studies
    total_loss_sm2, total_loss_sm1, total_loss_combined  (all averaged over slices)
    """
    sm2_logits = []
    sm1_logits = []
    label_list = []
    total_loss_sm2 = 0.0
    total_loss_sm1 = 0.0
    total_loss_total = 0.0
    num_sample = 0

    model.eval()
    for fea, data, labels in tqdm(val_loader, position=0):
        fea, data, labels = fea.float().cuda(), data.float().cuda(), labels.float().cuda()
        with torch.set_grad_enabled(False):
            logit, logit_help = model(fea, data)
            loss0 = criterion(logit, labels)
            loss1 = criterion(logit_help, labels)

        # logit shape: (B, 1, seq_len, 6) -- flatten to (B*seq_len, 6)
        sm2_logits.append(logit[:, 0].reshape(-1, 6).cpu().numpy())
        sm1_logits.append(logit_help[:, 0].reshape(-1, 6).cpu().numpy())
        label_list.append(labels.reshape(-1, 6).cpu().numpy())

        n = data.size(2)
        total_loss_sm2 += float(loss0.item()) * n
        total_loss_sm1 += float(loss1.item()) * n
        total_loss_total += float(loss0.item() + loss1.item()) * n
        num_sample += n

    sm2_logits = np.concatenate(sm2_logits, axis=0)
    sm1_logits = np.concatenate(sm1_logits, axis=0)
    gt = np.concatenate(label_list, axis=0)

    # Sigmoid in numpy (avoid moving back to GPU)
    sm2_probs = 1.0 / (1.0 + np.exp(-sm2_logits))
    sm1_probs = 1.0 / (1.0 + np.exp(-sm1_logits))

    return (sm2_probs, sm1_probs, gt,
            total_loss_sm2 / max(num_sample, 1),
            total_loss_sm1 / max(num_sample, 1),
            total_loss_total / max(num_sample, 1))


def train(smoke=False):

    n_folds = 1 if smoke else fold_num
    n_epochs = 2 if smoke else train_epoch

    kf = KFold(n_splits=fold_num, shuffle=True, random_state=48)
    all_df = pd.read_csv(rf'{csv_root}/train_meta_id_seriser.csv')
    StudyInstance = list(all_df['StudyInstance'].unique())
    print(len(StudyInstance))
    dict_ = get_train_dict()

    for s_fold, (train_idx, valid_idx) in enumerate(kf.split(StudyInstance)):
        print('fold ' + str(s_fold))

        if s_fold != fold_index and fold_index > 0:
            continue
        if smoke and s_fold >= n_folds:
            break

        batch_size = 128
        train_data = StackingDataset_study( dict_, X,y, train_idx, seq_len = seq_len, mode='train', Add_position=Add_position)
        train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=8, pin_memory=True, drop_last=True, shuffle=True)
        val_data = StackingDataset_study( dict_, X,y, valid_idx, seq_len = -1, mode='valid', Add_position=Add_position)
        val_loader = DataLoader(val_data, batch_size=1, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)

        model = SequenceModel(model_num = model_num, feature_dim = feature_dim, feature_num=feature_num,
                              lstm_layers = lstm_layers, hidden=hidden,
                              drop_out=drop_out,
                              Add_position = Add_position).cuda()

        print(model)
        optimizer = optim.Adam(model.parameters(), lr=3e-4)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [20, 30, 40], gamma=0.1)

        # Per-fold MetricLogger
        fold_root = os.path.join(model_save_dir, f'fold{s_fold}')
        logger = MetricLogger(fold_root, save_predictions=True)
        logger.write_metadata({
            'component': 'SequenceModel',
            'fold': s_fold,
            'fold_num': fold_num,
            'train_epoch': n_epochs,
            'lr_initial': 3e-4,
            'scheduler': 'MultiStepLR([20,30,40], gamma=0.1)',
            'optimizer': 'Adam',
            'loss': 'BCEWithLogits, weights=[2,1,1,1,1,1], sum(SM1+SM2)',
            'seq_len_train': seq_len,
            'Add_position': Add_position,
            'lstm_layers': lstm_layers,
            'hidden': hidden,
            'drop_out': drop_out,
            'model_num': model_num,
            'feature_dim': feature_dim,
            'feature_num': feature_num,
            'smoke_mode': smoke,
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        })

        best_score = 100
        for epoch in range(n_epochs):
            reset_system_metrics()
            epoch_start = time.time()
            running_loss = 0.0
            running_loss_sm2 = 0.0
            running_loss_sm1 = 0.0
            grad_acc = GradNormAccumulator()
            n_train_samples = 0
            model.train()
            for fea, data, labels in tqdm(train_loader, position=0):
                fea, data, labels = fea.float().cuda(), data.float().cuda(), labels.float().cuda()

                optimizer.zero_grad()
                with torch.set_grad_enabled(True):
                    fea, data, labels = fea.cuda(), data.cuda(), labels.cuda()
                    logit, logit_help = model(fea, data)
                    loss0 = criterion(logit, labels)
                    loss1 = criterion(logit_help, labels)

                    loss = loss0 + loss1
                    loss.backward()
                    grad_acc.record(model)
                    optimizer.step()

                running_loss += float(loss.item()) * data.shape[0]
                running_loss_sm2 += float(loss0.item()) * data.shape[0]
                running_loss_sm1 += float(loss1.item()) * data.shape[0]
                n_train_samples += data.shape[0]

            train_loss = running_loss / train_data.__len__()
            train_loss_sm2 = running_loss_sm2 / train_data.__len__()
            train_loss_sm1 = running_loss_sm1 / train_data.__len__()
            scheduler.step()

            # Full validation with per-class metrics
            sm2_probs, sm1_probs, gt, val_loss_sm2, val_loss_sm1, val_loss_total = \
                _collect_val_predictions(model, val_loader)
            val_loss = val_loss_sm2  # Legacy: original code only tracked SM2 loss

            epoch_time = time.time() - epoch_start

            sm2_metrics = compute_full_metrics(gt, sm2_probs, class_names=HEMORRHAGE_CLASSES)
            sm1_metrics = compute_full_metrics(gt, sm1_probs, class_names=HEMORRHAGE_CLASSES)

            print(model_save_dir)
            print(str(epoch), 'train_loss:{} val_loss:{} score:{}'.format(train_loss, val_loss, val_loss))
            log.write('fold: '+str(s_fold)+' '+str(epoch)+' train_loss:{} val_loss:{} score:{}'.format(train_loss, val_loss, val_loss))
            log.write('\n')

            # Build CSV row
            grad_stats = grad_acc.summary()
            sysm = system_metrics()
            csv_row = {
                'Epoch': epoch,
                'LR': round(optimizer.param_groups[0]['lr'], 8),
                'EpochTime_s': round(epoch_time, 2),
                'TrainLoss_total': round(float(train_loss), 6),
                'TrainLoss_SM2': round(train_loss_sm2, 6),
                'TrainLoss_SM1': round(train_loss_sm1, 6),
                'ValLoss_total': round(val_loss_total, 6),
                'ValLoss_SM2': round(val_loss_sm2, 6),
                'ValLoss_SM1': round(val_loss_sm1, 6),
                'GradNorm_mean': round(grad_stats['mean'], 6),
                'GradNorm_std': round(grad_stats['std'], 6),
                'GradNorm_max': round(grad_stats['max'], 6),
                'GPUmem_peak_GB': round(sysm['gpu_peak_mem_GB'], 3),
                'n_train_samples': n_train_samples,
                'n_val_slices': int(gt.shape[0]),
            }
            for k, v in flatten_for_csv(sm2_metrics, prefix='SM2_').items():
                csv_row[k] = round(v, 6) if isinstance(v, float) else v
            for k, v in flatten_for_csv(sm1_metrics, prefix='SM1_').items():
                csv_row[k] = round(v, 6) if isinstance(v, float) else v

            structured = {
                'epoch': epoch,
                'fold': s_fold,
                'sm2': sm2_metrics,
                'sm1': sm1_metrics,
                'losses': {
                    'train_total': float(train_loss),
                    'train_sm2': train_loss_sm2,
                    'train_sm1': train_loss_sm1,
                    'val_total': val_loss_total,
                    'val_sm2': val_loss_sm2,
                    'val_sm1': val_loss_sm1,
                },
                'grad_norm': grad_stats,
                'system': sysm,
            }

            # Save SM2 + SM1 predictions for post-hoc analysis
            npz_path = os.path.join(fold_root, f'epoch_{epoch:03d}_val_predictions.npz')
            os.makedirs(fold_root, exist_ok=True)
            np.savez_compressed(npz_path,
                                sm2_probs=sm2_probs.astype(np.float32),
                                sm1_probs=sm1_probs.astype(np.float32),
                                labels=gt.astype(np.float32))

            logger.log_epoch(epoch, csv_row, full_metrics=structured, gt=None, pred=None)

            # Track best by SM2 val AUC any
            sm2_any_auc = sm2_metrics['per_class']['any']['AUC']
            if logger.update_best(epoch, sm2_any_auc,
                                   extra={'macro_AUC_SM2': sm2_metrics['macro']['macro_AUC']}):
                torch.save(model.state_dict(),
                           os.path.join(model_save_dir, f'fold_{s_fold}_best_auc.pt'))

            # Legacy best-by-loss checkpoint (original behaviour preserved)
            if best_score > val_loss:
                best_score = val_loss
                print('save max score!!!!!!!!!!!!')
                log.write('save max score!!!!!!!!!!!!')
                log.write('\n')
                torch.save(model.state_dict(), os.path.join(model_save_dir,'fold_' + str(s_fold) + '.pt'))

            gc.collect()

def valid():

    kf = KFold(n_splits=fold_num, shuffle=True, random_state=48)
    all_df = pd.read_csv(rf'{csv_root}/train_meta_id_seriser.csv')
    StudyInstance = list(all_df['StudyInstance'].unique())
    print(len(StudyInstance))
    dict_ = get_train_dict()

    logit_list = []
    label_list = []
    for s_fold, (train_idx, valid_idx) in enumerate(kf.split(StudyInstance)):
        val_data = StackingDataset_study(dict_, X, y, valid_idx, seq_len=-1, mode='valid', reverse=True, Add_position=Add_position)
        val_loader = DataLoader(val_data, batch_size=1, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)

        model = SequenceModel(model_num=model_num, feature_dim=feature_dim, feature_num=feature_num,
                              lstm_layers=lstm_layers, hidden=hidden,
                              drop_out=drop_out,
                              Add_position=Add_position).cuda()

        print('fold ' + str(s_fold))
        model.load_state_dict(torch.load(os.path.join(model_save_dir,'fold_'+str(s_fold) + '.pt'), weights_only=False))
        model.eval()
        for fea, data, labels in tqdm(val_loader, position=0):
            fea, data, labels = fea.float().cuda(), data.float().cuda(), labels.float().cuda()

            with torch.set_grad_enabled(False):
                logit,_ = model(fea, data)
                logit_list.append(logit)
                label_list.append(labels)
        print('===============================================================================================')

    running_loss = 0
    num_sample =0

    for logit, labels in zip(logit_list, label_list):
        loss = criterion(logit, labels)
        running_loss += loss.item() * logit.size(2)
        num_sample += logit.size(2)
    val_loss = running_loss / num_sample
    print(val_loss)
    log.write(str(val_loss))
    log.write('\n')

    logit_list_flip = []
    label_list_flip = []
    for s_fold, (train_idx, valid_idx) in enumerate(kf.split(StudyInstance)):
        val_data = StackingDataset_study(dict_, X, y, valid_idx, seq_len=-1, mode='valid', reverse=False, Add_position=Add_position)
        val_loader = DataLoader(val_data, batch_size=1, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)

        model = SequenceModel(model_num=model_num, feature_dim=feature_dim, feature_num=feature_num,
                              lstm_layers=lstm_layers, hidden=hidden,
                              drop_out=drop_out,
                              Add_position=Add_position).cuda()

        print('fold ' + str(s_fold))
        model.load_state_dict(torch.load(os.path.join(model_save_dir, 'fold_' + str(s_fold) + '.pt'), weights_only=False))
        model.eval()
        for fea, data, labels in tqdm(val_loader, position=0):
            fea, data, labels = fea.float().cuda(), data.float().cuda(), labels.float().cuda()

            with torch.set_grad_enabled(False):
                logit,_ = model(fea, data)
                logit_list_flip.append(logit)
                label_list_flip.append(labels)

        print('===============================================================================================')

    running_loss = 0
    num_sample =0
    for logit, labels in zip(logit_list_flip, label_list_flip):
        loss = criterion(logit, labels)
        running_loss += loss.item() * logit.size(2)
        num_sample += logit.size(2)
    val_loss = running_loss / num_sample
    print(val_loss)
    log.write(str(val_loss))
    log.write('\n')

    running_loss = 0
    num_sample =0

    for logit,logit_flip, labels, labels_flip in zip(logit_list,logit_list_flip, label_list, label_list_flip):
        logit = logit.cpu().numpy()
        logit_flip = logit_flip.cpu().numpy()
        logit = (logit + logit_flip[:,:,::-1,:]) / 2.0
        logit = torch.from_numpy(logit)
        logit = logit.float().cuda()

        loss = criterion(logit, labels)
        running_loss += loss.item() * logit.size(2)
        num_sample += logit.size(2)

    val_loss = running_loss / num_sample
    print(val_loss)
    log.write('final!!!!!!!!!!!!')
    log.write('\n')
    log.write(str(val_loss))
    log.write('\n')

def inference():
    predicts_list = []
    for s_fold in range(fold_num):
        running_loss = 0
        num_sample =0

        test_id_dict = get_test_dict()
        dataset = StackingDataset_study(test_id_dict, X_test, None, None, seq_len=-1, mode='test', reverse=False, Add_position=Add_position)
        val_loader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)

        model = SequenceModel(model_num=model_num, feature_dim=feature_dim, feature_num=feature_num,
                              lstm_layers=lstm_layers, hidden=hidden,
                              drop_out=drop_out,
                              Add_position=Add_position).cuda()

        print('fold ' + str(s_fold))
        model.load_state_dict(torch.load(os.path.join(model_save_dir,'fold_'+str(s_fold) + '.pt'), weights_only=False))
        model.eval()

        filenames_list = []
        for filenames, inputs_fea, inputs in tqdm(val_loader, position=0):
            filenames_list.extend(filenames)

            inputs = inputs.float().cuda()
            inputs_fea= inputs_fea.float().cuda()

            with torch.set_grad_enabled(False):
                logit,_ = model(inputs_fea, inputs)
                logit = torch.sigmoid(logit)

            logit = logit.cpu().numpy()
            logit = logit.reshape([-1, 6])

            if num_sample != 0:
                predicts = np.vstack((predicts, logit))
            else:
                predicts = logit
            num_sample += inputs.size(2)

        print(predicts.shape)
        print(num_sample)
        predicts_list.append(predicts)

    final = np.mean(predicts_list,axis=0)
    predicts_list = []
    for s_fold in range(5):
        running_loss = 0
        num_sample =0

        test_id_dict = get_test_dict()
        dataset = StackingDataset_study(test_id_dict, X_test, None, None, seq_len=-1, mode='test', reverse=True, Add_position=Add_position)
        val_loader = DataLoader(dataset, batch_size=1, num_workers=8, pin_memory=True, drop_last=False, shuffle=False)

        model = SequenceModel(model_num=model_num, feature_dim=feature_dim, feature_num=feature_num,
                              lstm_layers=lstm_layers, hidden=hidden,
                              drop_out=drop_out,
                              Add_position=Add_position).cuda()

        print('fold ' + str(s_fold))
        model.load_state_dict(torch.load(os.path.join(model_save_dir,'fold_'+str(s_fold) + '.pt'), weights_only=False))
        model.eval()

        filenames_list = []
        for filenames, inputs_fea, inputs in tqdm(val_loader, position=0):
            filenames_list.extend(filenames)

            inputs = inputs.float().cuda()
            inputs_fea= inputs_fea.float().cuda()

            with torch.set_grad_enabled(False):
                logit,_ = model(inputs_fea, inputs)
                logit = torch.sigmoid(logit)

            logit = logit.cpu().numpy()
            logit = logit.reshape([-1, 6])
            logit = logit[::-1, :]

            if num_sample != 0:
                predicts = np.vstack((predicts, logit))
            else:
                predicts = logit
            num_sample += inputs.size(2)

        print(predicts.shape)
        print(num_sample)
        predicts_list.append(predicts)

    final_flip = np.mean(predicts_list,axis=0)
    final = (final + final_flip)/2.0

    filenames_list = list(np.asarray(filenames_list).reshape([-1]))
    test_df = pd.DataFrame()
    test_df['filename'] = filenames_list

    test_df = test_df.join(pd.DataFrame(final, columns=[
        'any', 'epidural', 'intraparenchymal', 'intraventricular', 'subarachnoid', 'subdural'
    ]))

    # Unpivot table, i.e. wide (N x 6) to long format (6N x 1)
    test_df = test_df.melt(id_vars=['filename'])
    # Combine the filename column with the variable column
    test_df['ID'] = test_df.filename.apply(lambda x: x.replace('.dcm', '')) + '_' + test_df.variable
    test_df['Label'] = test_df['value']
    test_df[['ID', 'Label']].to_csv(os.path.join(model_save_dir,'submission_tta.csv'), index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true',
                        help='Smoke test: 1 fold, 2 epochs')
    parser.add_argument('--skip-valid', action='store_true',
                        help='Skip valid()/inference() phases (useful for smoke test)')
    args = parser.parse_args()

    fold_index = -1
    fold_num = 5
    Add_position = True
    lstm_layers = 2
    seq_len = 24
    hidden = 96
    drop_out = 0.5
    train_epoch = 40
    # train_epoch = 1
    class_num = 6

    train(smoke=args.smoke)
    if not args.skip_valid and not args.smoke:
        valid()
        inference()

