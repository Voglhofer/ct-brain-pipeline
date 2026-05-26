import os
import sys
import time
import pandas as pd
import gc
import cv2
import csv
import json
import random
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
from torch.optim.lr_scheduler import ReduceLROnPlateau,MultiStepLR
import torch.utils.data

import torch.utils.data as data
from net.models import *
from dataset.dataset import *
from tuils.tools import *
from tuils.lrs_scheduler import WarmRestart, warm_restart, AdamW, RAdam
from tuils.loss_function import *
import torch.nn.functional as F
from collections import OrderedDict
import warnings
warnings.filterwarnings('ignore')
torch.manual_seed(1992)
torch.cuda.manual_seed(1992)
np.random.seed(1992)
random.seed(1992)
from PIL import ImageFile
import sklearn
import copy
torch.backends.cudnn.benchmark = True
import argparse

# --- Metric instrumentation: passive logging only, does NOT change training ---
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
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

def epochVal(model, dataLoader, loss_cls, c_val, val_batch_size):
    model.eval ()
    lossValNorm = 0
    valLoss = 0

    outGT = torch.FloatTensor().cuda()
    outPRED = torch.FloatTensor().cuda()
    for i, (input, target) in enumerate (dataLoader):
        if i == 0:
            ss_time = time.time()
        print(str(i) + '/' + str(int(len(c_val)/val_batch_size)) + '     ' + str((time.time()-ss_time)/(i+1)), end='\r')
        target = target.view(-1, 6).contiguous().cuda()
        outGT = torch.cat((outGT, target), 0)
        varInput = input
        varTarget = target.contiguous().cuda()
        varOutput = model(varInput)
        lossvalue = loss_cls(varOutput, varTarget)
        valLoss = valLoss + lossvalue.item()
        varOutput = varOutput.sigmoid()

        outPRED = torch.cat((outPRED, varOutput.data), 0)
        lossValNorm += 1

    valLoss = valLoss / lossValNorm

    auc = computeAUROC(outGT, outPRED, 6)
    auc = [round(x, 4) for x in auc]
    loss_list, loss_sum = weighted_log_loss(outPRED, outGT)

    # Return raw arrays in addition to legacy outputs so the caller can
    # compute the full metric suite without re-running inference.
    outGT_np = outGT.cpu().numpy()
    outPRED_np = outPRED.cpu().numpy()

    return valLoss, auc, loss_list, loss_sum, outGT_np, outPRED_np

def train(model_name, image_size, smoke=False):

    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    # Legacy human-readable log; preserved for backwards compatibility.
    legacy_log_path = os.path.join(snapshot_path, 'log_legacy.csv')
    if not os.path.isfile(legacy_log_path):
        with open(legacy_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Epoch', 'Learning rate', 'Time', 'Train Loss', 'Val Loss'])
    df_all = pd.read_csv(csv_path)

    kfold_path_train = '../data/fold_5_by_study/'
    kfold_path_val = '../data/fold_5_by_study_image/'

    n_folds = 1 if smoke else 5
    trMaxEpoch = 2 if smoke else 80

    for num_fold in range(n_folds):
        print('fold_num:',num_fold)

        with open(legacy_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([num_fold])

        f_train = open(kfold_path_train + 'fold' + str(num_fold) + '/train.txt', 'r')
        f_val = open(kfold_path_val + 'fold' + str(num_fold) + '/val.txt', 'r')
        c_train = f_train.readlines()
        c_val = f_val.readlines()
        f_train.close()
        f_val.close()
        c_train = [s.replace('\n', '') for s in c_train]
        c_val = [s.replace('\n', '') for s in c_val]

        if smoke:
            c_train = c_train[:100]
            c_val = c_val[:50]

        print('train dataset study num:', len(c_train), '  val dataset image num:', len(c_val))
        with open(legacy_log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['train dataset:', len(c_train), '  val dataset:', len(c_val)])
            writer.writerow(['train_batch_size:', train_batch_size, 'val_batch_size:', val_batch_size])

        train_transform, val_transform = generate_transforms(image_size)
        train_loader, val_loader = generate_dataset_loader(df_all, c_train, train_transform, train_batch_size, c_val, val_transform, val_batch_size, workers)

        # Light validation set: fixed deterministic subset for per-epoch validation
        # (full val runs only every 5 epochs because it's expensive).
        # Gives 80 datapoints for the learning curve in the thesis instead of 16.
        LIGHT_VAL_N = 50 if smoke else 5000
        c_val_light = c_val[:LIGHT_VAL_N]
        light_val_dataset = RSNA_Dataset_val_by_study_context(df_all, c_val_light, val_transform)
        light_val_loader = torch.utils.data.DataLoader(
            light_val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False,
        )

        model = eval(model_name+'()')
        model = model.cuda()

        optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00002)
        scheduler = WarmRestart(optimizer, T_max=5, T_mult=1, eta_min=1e-5)
        model = torch.nn.DataParallel(model)
        loss_cls = torch.nn.BCEWithLogitsLoss(pos_weight = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda())

        # Per-fold metric logger writes to <snapshot_path>/fold<N>/
        fold_root = os.path.join(snapshot_path, f'fold{num_fold}')
        logger = MetricLogger(fold_root, save_predictions=True)
        logger.write_metadata({
            'backbone': model_name,
            'image_size': image_size,
            'train_batch_size': train_batch_size,
            'val_batch_size': val_batch_size,
            'workers': workers,
            'fold': num_fold,
            'n_train_studies': len(c_train),
            'n_val_images': len(c_val),
            'n_light_val_images': len(c_val_light),
            'light_val_policy': 'first LIGHT_VAL_N images of val.txt, evaluated every epoch',
            'optimizer': 'Adam',
            'lr_initial': 0.0005,
            'weight_decay': 0.00002,
            'betas': [0.9, 0.999],
            'scheduler': 'WarmRestart(T_max=5, T_mult=1, eta_min=1e-5)',
            'epochs': trMaxEpoch,
            'loss': 'BCEWithLogitsLoss(pos_weight=[1]*6)',
            'seed': 1992,
            'smoke_mode': smoke,
            'started_at': time.strftime('%Y-%m-%dT%H:%M:%S'),
        })

        for epochID in range (0, trMaxEpoch):
            epochID = epochID + 0

            reset_system_metrics()
            start_time = time.time()
            model.train()
            trainLoss = 0
            lossTrainNorm = 10

            if epochID < 10:
                pass
            elif epochID < 80:
                if epochID != 10:
                    scheduler.step()
                    scheduler = warm_restart(scheduler, T_mult=2)
            else:
                optimizer.param_groups[0]['lr'] = 1e-5

            grad_acc = GradNormAccumulator()
            batch_times = []
            n_samples_seen = 0
            last_batch_end = time.time()

            for batchID, (input, target) in enumerate (train_loader):
                if batchID == 0:
                    ss_time = time.time()

                print(str(batchID) + '/' + str(int(len(c_train)/train_batch_size)) + '     ' + str((time.time()-ss_time)/(batchID+1)), end='\r')
                varInput = input
                target = target.view(-1, 6).contiguous().cuda()
                varTarget = target.contiguous().cuda()
                varOutput = model(varInput)
                lossvalue = loss_cls(varOutput, varTarget)
                trainLoss = trainLoss + lossvalue.item()
                lossTrainNorm = lossTrainNorm + 1

                lossvalue.backward()
                # Record gradient L2-norm BEFORE optimizer.step()/zero_grad()
                # so the gradient is still in .grad. This does not modify grads.
                grad_acc.record(model)
                optimizer.step()
                optimizer.zero_grad()

                now = time.time()
                batch_times.append(now - last_batch_end)
                last_batch_end = now
                n_samples_seen += target.size(0)
                del lossvalue

            trainLoss = trainLoss / lossTrainNorm

            valLoss = float('nan')
            auc = [float('nan')] * 6
            loss_list = [float('nan')] * 6
            loss_sum = float('nan')
            outGT_np = None
            outPRED_np = None

            # --- Light validation EVERY epoch (per-epoch learning curve) ---
            # Uses fixed deterministic subset of val.txt, ~3% of full val set.
            # Adds ~2-3 min per epoch on RTX 5090, total ~+3 hours across the run.
            light_val_loss, light_auc, _, _, light_outGT_np, light_outPRED_np = epochVal(
                model, light_val_loader, loss_cls, c_val_light, val_batch_size)

            # --- Full validation every 5 epochs (best-model selection, unchanged) ---
            valLoss = float('nan')
            auc = [float('nan')] * 6
            loss_list = [float('nan')] * 6
            loss_sum = float('nan')
            outGT_np = None
            outPRED_np = None

            run_val = (epochID+1) % 5 == 0 or epochID > 79 or epochID == 0
            if smoke:
                run_val = True
            if run_val:
                valLoss, auc, loss_list, loss_sum, outGT_np, outPRED_np = epochVal(
                    model, val_loader, loss_cls, c_val, val_batch_size)

            epoch_time = time.time() - start_time

            if (epochID+1) % 5 == 0 or epochID > 79 or smoke:
                torch.save({'epoch': epochID + 1, 'state_dict': model.state_dict(), 'valLoss': valLoss}, snapshot_path + '/model_epoch_' + str(epochID) + '_' + str(num_fold) + '.pth')

            # --- Full metric computation (only on val epochs) ---
            full_metrics = None
            csv_row = {
                'Epoch': epochID,
                'LR': round(optimizer.state_dict()['param_groups'][0]['lr'], 8),
                'EpochTime_s': round(epoch_time, 2),
                'TrainLoss': round(trainLoss, 5),
                # Light val: every epoch (deterministic subset)
                'LightValLoss': round(light_val_loss, 5),
                'LightValAUC_any': round(light_auc[0], 5),
                'LightValAUC_epi': round(light_auc[1], 5),
                'LightValAUC_ipa': round(light_auc[2], 5),
                'LightValAUC_ive': round(light_auc[3], 5),
                'LightValAUC_sah': round(light_auc[4], 5),
                'LightValAUC_sdh': round(light_auc[5], 5),
                # Full val: every 5 epochs (empty cells when not run)
                'ValLoss': round(valLoss, 5) if np.isfinite(valLoss) else '',
                'ValLegacyLossSum': loss_sum,
                'ValLegacyLossPerClass': str(loss_list),
                'ValLegacyAUC': str(auc),
            }

            # Light val: compute the full per-class metric suite each epoch
            light_full_metrics = compute_full_metrics(light_outGT_np, light_outPRED_np,
                                                     class_names=HEMORRHAGE_CLASSES)
            light_flat = flatten_for_csv(light_full_metrics, prefix='Light_')
            for k, v in light_flat.items():
                csv_row[k] = round(v, 6) if isinstance(v, float) else v

            # Gradient + system metrics
            grad_stats = grad_acc.summary()
            csv_row['GradNorm_mean'] = round(grad_stats['mean'], 6)
            csv_row['GradNorm_std'] = round(grad_stats['std'], 6)
            csv_row['GradNorm_max'] = round(grad_stats['max'], 6)
            sysm = system_metrics()
            csv_row['GPUmem_peak_GB'] = round(sysm['gpu_peak_mem_GB'], 3)
            csv_row['GPUmem_reserved_GB'] = round(sysm['gpu_reserved_mem_GB'], 3)
            csv_row['BatchTime_mean_s'] = round(float(np.mean(batch_times)) if batch_times else 0.0, 4)
            csv_row['SamplesPerSec'] = round(n_samples_seen / epoch_time, 2) if epoch_time > 0 else 0.0

            if outGT_np is not None and outPRED_np is not None:
                full_metrics = compute_full_metrics(outGT_np, outPRED_np,
                                                   class_names=HEMORRHAGE_CLASSES)
                flat = flatten_for_csv(full_metrics)
                # Round floats for readability
                for k, v in flat.items():
                    if isinstance(v, float):
                        csv_row[k] = round(v, 6)
                    else:
                        csv_row[k] = v

                # Best-model checkpoint based on val AUC for "any" hemorrhage
                any_auc = full_metrics['per_class']['any']['AUC']
                if logger.update_best(epochID, any_auc, extra={'macro_AUC': full_metrics['macro']['macro_AUC']}):
                    torch.save({'epoch': epochID + 1,
                                'state_dict': model.state_dict(),
                                'valLoss': valLoss,
                                'val_AUC_any': any_auc},
                               os.path.join(snapshot_path, f'model_best_{num_fold}.pth'))

            # Log via MetricLogger (handles CSV + JSON + npz for full val)
            logger.log_epoch(epochID, csv_row, full_metrics=full_metrics,
                             gt=outGT_np, pred=outPRED_np)

            # Always dump light val predictions for the learning-curve plots
            np.savez_compressed(
                os.path.join(fold_root, f'epoch_{epochID:03d}_lightval_predictions.npz'),
                outGT=light_outGT_np.astype(np.float32),
                outPRED=light_outPRED_np.astype(np.float32),
            )

            # Legacy log preserves the exact original format
            legacy_result = [epochID,
                             round(optimizer.state_dict()['param_groups'][0]['lr'], 6),
                             round(epoch_time, 0),
                             round(trainLoss, 5),
                             round(valLoss, 5),
                             'auc:', auc,
                             'loss:', loss_list,
                             loss_sum]
            print(legacy_result)
            with open(legacy_log_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(legacy_result)

        del model

def valid_snapshot(model_name, image_size):
    dir = r'./DenseNet121_change_avg_256'
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    header = ['Epoch', 'Learning rate', 'Time', 'Train Loss', 'Val Loss']

    if not os.path.isfile(snapshot_path + '/log.csv'):
        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
    df_all = pd.read_csv(csv_path)

    kfold_path_val = '../data/fold_5_by_study_image/'
    loss_cls = torch.nn.BCEWithLogitsLoss(pos_weight=torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]).cuda())
    for num_fold in range(5):
        print('fold_num:', num_fold)

        ckpt = r'model_epoch_best_'+str(num_fold)+'.pth'
        ckpt = os.path.join(dir,ckpt)

        with open(snapshot_path + '/log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([num_fold])

        f_val = open(kfold_path_val + 'fold' + str(num_fold) + '/val.txt', 'r')
        c_val = f_val.readlines()
        f_val.close()
        c_val = [s.replace('\n', '') for s in c_val]

        print('  val dataset image num:', len(c_val))

        val_transform = albumentations.Compose([
            albumentations.Resize(image_size, image_size),
            albumentations.Normalize(mean=(0.456, 0.456, 0.456), std=(0.224, 0.224, 0.224), max_pixel_value=255.0,
                                     p=1.0)
        ])

        val_dataset = RSNA_Dataset_val_by_study_context(df_all, c_val, val_transform)

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=workers,
            pin_memory=True,
            drop_last=False)

        model = eval(model_name + '()')
        model = model.cuda()
        model = torch.nn.DataParallel(model)

        if ckpt is not None:
            print(ckpt)
            model.load_state_dict(torch.load(ckpt, map_location=lambda storage, loc: storage, weights_only=False)["state_dict"])

        valLoss, auc, loss_list, loss_sum, _, _ = epochVal(model, val_loader, loss_cls, c_val, val_batch_size)

        result = [round(valLoss, 5),
                  'auc:', auc,
                  'loss:', loss_list,
                  loss_sum]

        with open(ckpt + '_log.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(result)
        print(result)


if __name__ == '__main__':
    csv_path = '../data/stage1_train_cls.csv'
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-backbone", "--backbone", type=str, default='DenseNet121_change_avg', help='backbone')
    parser.add_argument("-img_size", "--Image_size", type=int, default=256, help='image_size')
    parser.add_argument("-tbs", "--train_batch_size", type=int, default=32, help='train_batch_size')
    parser.add_argument("-vbs", "--val_batch_size", type=int, default=32, help='val_batch_size')
    parser.add_argument("-save_path", "--model_save_path", type=str,
                        default='DenseNet169_change_avg', help='epoch')
    parser.add_argument("--smoke", action='store_true',
                        help='Smoke test: 1 fold, 2 epochs, 100 train / 50 val samples')
    args = parser.parse_args()

    Image_size = args.Image_size
    train_batch_size = args.train_batch_size
    val_batch_size = args.val_batch_size
    workers = 24
    backbone = args.backbone
    print(backbone)
    print('image size:', Image_size)
    print('train batch size:', train_batch_size)
    print('val batch size:', val_batch_size)
    print('smoke mode:', args.smoke)
    save_path = args.model_save_path.replace('\n', '').replace('\r', '')
    if args.smoke:
        save_path = save_path + '_smoke'
    snapshot_path = 'data_test/' + save_path
    train(backbone, Image_size, smoke=args.smoke)
    # valid_snapshot(backbone, Image_size)
