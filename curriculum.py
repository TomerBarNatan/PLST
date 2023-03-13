

from utils import load_model, get_batch, loss_calc
from model.unet import UNet2D
import torch
import torch.optim as optim
from tqdm import tqdm
import wandb
import json
from metric_utils import get_sdice, get_dice
import os
import pandas as pd
from print_seg import print_seg
import math
best_metric = -1
low_source_metric = 1.1

prev_d_score = 0


def curriculum(model_path, train_loader, target_loader, val_ds, test_ds, val_ds_source, args,
                             config):
    print("curriculum learning")
    model = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)
    model = load_model(model, model_path, config.msm)
    torch.save(model.state_dict(), config.exp_dir / f'best_model.pth')
    torch.save(model.state_dict(), config.exp_dir / f'latest_model.pth')
    model.eval()
    model.to(args.gpu)
    if config.parallel_model:
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])

    train_loader.dataset.yield_id = True
    target_loader.dataset.yield_id = True
    train_loader_iter = iter(train_loader)
    target_loader_iter = iter(target_loader)
    if config.msm:
        optimizer = optim.Adam(model.parameters(),
                               lr=1e-6, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=1e-6)
    iterations = 50
    epochs = 15
    for i in tqdm(range(iterations), "iteration", position=0, leave=True):
        if i > 20:
            epochs = 15
        pseudo_labeling_after_step(model, config, i, iterations, test_ds, val_ds_source, val_ds, args)
        model_path = config.exp_dir / f'latest_model.pth'
        labels_generator = UNet2D(config.n_channels, n_chans_out=config.n_chans_out)
        labels_generator = load_model(labels_generator, model_path, config.msm)
        labels_generator.eval()
        for epoch in tqdm(range(epochs), desc="epoch", position=0, leave=True):
            optimizer.zero_grad()

            source_train_batch, target_batch, source_size, target_size = get_filtered_batches(train_loader, train_loader_iter, target_loader, target_loader_iter, i, iterations, args)
            if target_size > 0:
                target_images, _, target_ids, target_slice_nums = target_batch
                _, target_labels_preds = labels_generator(target_images)
                target_labels = torch.argmax(target_labels_preds, dim=1)
                _, target_preds = model(target_images.to(args.gpu))
                target_loss = loss_calc(target_preds, target_labels, gpu=args.gpu)
                del target_preds
                del target_images
            else:
                target_loss = 0
            if source_size > 0:
                source_train_images, source_train_labels, source_train_ids, source_train_slice_nums = source_train_batch
                _, source_preds = model(source_train_images.to(args.gpu))
                source_loss = loss_calc(source_preds, source_train_labels, gpu=args.gpu)
                del source_train_images
                del source_preds
            else:
                source_loss = 0
            total_loss = source_loss + target_loss
            total_loss.backward()
            optimizer.step()
            del source_loss
            del target_loss
        torch.save(model.state_dict(), config.exp_dir / f'latest_model.pth')
    
    pseudo_labeling_after_step(model, config, iterations - 1, iterations, test_ds, val_ds_source, val_ds, args)
    print("Finished Updating PL!")


def get_filtered_batches(train_loader, train_loader_iter, target_loader, target_loader_iter, iteration, max_iterations, args):
    df = pd.read_csv(f"{os.getcwd()}/dataset/{args.algo}/{args.source}_{args.target}.csv")
    source_batch_size, target_batch_size, source_df, target_df = get_curriculum_policy_params(iteration, max_iterations, 8, df)
    source_new_batch = [torch.tensor([])] * 4
    data_size = 0
    while data_size < source_batch_size:
        if source_df.empty:
            print(f"Source df is empty! Curr data size {data_size}")
            break
        source_batch = get_batch(train_loader, train_loader_iter)
        source_new_batch, data_size, source_df= filter_batches(source_batch, source_new_batch, source_df, data_size, source_batch_size)
    source_new_batch[0] = torch.unsqueeze(source_new_batch[0], dim = 1)
    source_new_batch[1] = torch.unsqueeze(source_new_batch[1], dim = 1)
    
    target_new_batch = [torch.tensor([])] * 4
    data_size = 0
    while data_size < target_batch_size:
        if target_df.empty:
            print(f"Target df is empty! Curr data size {data_size}")
            break
        target_batch = get_batch(target_loader, target_loader_iter)
        target_new_batch, data_size, target_df = filter_batches(target_batch, target_new_batch, target_df, data_size, target_batch_size)

    target_new_batch[0] = torch.unsqueeze(target_new_batch[0], dim = 1)
    target_new_batch[1] = torch.unsqueeze(target_new_batch[1], dim = 1)
    return source_new_batch, target_new_batch, source_batch_size, target_batch_size


def get_curriculum_policy_params(iteration, max_iterations, bach_size, df: pd.DataFrame):
    target_batch_size =int((bach_size + (iteration/max_iterations) * bach_size) / 2)    
    source_batch_size = bach_size - target_batch_size

    all_target_df = df[df["label"] == 1]
    all_source_df = df[df["label"] == 0]

    source_precentage = math.ceil((all_source_df.shape[0] / 10))
    source_df = all_source_df.tail(source_precentage)

    target_precentage = math.ceil((10 + 90 * (iteration / max_iterations)) * (all_target_df.shape[0] / 100))
    target_df = all_target_df.head(target_precentage)

    return source_batch_size, target_batch_size, source_df, target_df

def filter_batches(batch, new_batch, df, data_size, max_data):
    for i in range(len(batch[0])):
        if data_size >= max_data:
            return new_batch, data_size, df
        if not df[(df["id"] == batch[2][i].item()) & (df["slice_num"] == batch[3][i].item())].empty:
            if data_size == 0:
                for j in range(4): new_batch[j] = batch[j][i]
            else:
                new_batch[0] = torch.cat((new_batch[0], batch[0][i]))
                new_batch[1] = torch.cat((new_batch[1], batch[1][i]))
                new_batch[2] = torch.cat((new_batch[2].view(-1,1), batch[2][i].view(1,1))).T[0]
                new_batch[3] = torch.cat((new_batch[3].view(-1,1), batch[3][i].view(1,1))).T[0]
            df = df.loc[~((df["id"] == batch[2][i].item()) & (df["slice_num"] == batch[3][i].item()))]
            data_size += 1
    return new_batch, data_size, df



def pseudo_labeling_after_step(model, config, step_num, epochs, test_ds, val_ds_source, val_ds, args):
    global best_metric
    global low_source_metric
    global prev_d_score
    if step_num % 10 == 0 and step_num != 0:
        if config.msm:
            dice1, sdice1 = get_dice(model, val_ds, args.gpu, config)
            main_metric = dice1
        else:
            dice1, sdice1 = get_sdice(model, val_ds, args.gpu, config)
            main_metric = sdice1
        wandb.log({f'pseudo_labeling_dice/val': dice1, f'pseudo_labeling_sdice/val': sdice1}, step=step_num)
        print(f'pseudo_labeling_dice is ', dice1)
        print(f'pseudo_labeling_sdice is ', sdice1)
        print('pseudo_labeling taking snapshot ...')

        if main_metric > best_metric:
            best_metric = main_metric
            print("new best metric!")
            torch.save(model.state_dict(), config.exp_dir / f'best_model.pth')

        torch.save(model.state_dict(), config.exp_dir / f'pseudo_labeling_model.pth')
    if step_num == 0 or step_num == epochs - 1:

        title = 'end' if step_num != 0 else 'start'
        scores = {}
        if config.msm:
            dice_test, sdice_test = get_dice(model, test_ds, args.gpu, config)
        else:
            dice_test, sdice_test = get_sdice(model, test_ds, args.gpu, config)

        scores[f'pseudo_labeling_dice_{title}/test'] = dice_test
        scores[f'pseudo_labeling_sdice_{title}/test'] = sdice_test
        print(f"dice {title} is: {dice_test}")
        print(f"sdice {title} is: {sdice_test}")
        if step_num != 0:
            model.load_state_dict(torch.load(config.exp_dir / f'best_model.pth', map_location='cpu'))
            if config.msm:
                dice_test_best, sdice_test_best = get_dice(model, test_ds, args.gpu, config)
            else:
                dice_test_best, sdice_test_best = get_sdice(model, test_ds, args.gpu, config)
            scores[f'dice_{title}/test_best'] = dice_test_best
            scores[f'sdice_{title}/test_best'] = sdice_test_best

        wandb.log(scores, step=step_num)
        json.dump(scores, open(config.exp_dir / f'scores_{title}.json', 'w'))
