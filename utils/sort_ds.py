from datasets.cc359_dataset import CC359Ds
from datasets.msm_dataset import MultiSiteMri
from dpipe.io import load
from torch.utils import data
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
from trainers.classification import site_classification

def create_sorted_dataset(model_path, args, config):
    if args.algo is None:
        raise("Args missing algo name!")
    if not args.msm:
        source_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/train_ids.json')[:config.data_len],
                            site=args.source)

        target_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.target}/train_ids.json')[:config.data_len],
                            site=args.target)
        val_ds_source = CC359Ds(load(f'{config.base_splits_path}/site_{args.source}/val_ids.json'), site=args.source,
                                    yield_id=True, slicing_interval=1)
        test_ds = CC359Ds(load(f'{config.base_splits_path}/site_{args.target}/test_ids.json'), site=args.target,
                            yield_id=True, slicing_interval=1)
    else:
        assert args.source == args.target
        print(f"source/target is {args.source}")
        source_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/train_ids.json'))
        target_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.target}/train_ids.json'))

        val_ds_source = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.source}t/val_ids.json'), yield_id=True,
                                     test=True)
        test_ds = MultiSiteMri(load(f'{config.base_splits_path}/site_{args.target}/test_ids.json'), yield_id=True,
                               test=True)
    source_loader = data.DataLoader(source_ds, batch_size=1, shuffle=False)
    target_loader = data.DataLoader(target_ds, batch_size=1, shuffle=False)
    source_loader.dataset.yield_id = True
    target_loader.dataset.yield_id = True
    classifier, model = site_classification(model_path, source_loader, target_loader, test_ds, val_ds_source, args, config, return_classifier=True, msm=args.msm)
    iter_source = iter(source_loader)
    source_res = {"confidence": [], "id": [], "slice_num": []}
    target_res = {"confidence": [], "id": [], "slice_num": []}

    for image, _, id, slice_num in tqdm(iter_source, desc='running source loader', position=0, leave=True):
        _, _, bottleneck = model(image.to(args.gpu))
        model_op = classifier(bottleneck.to(args.gpu))
        soft_pred = F.softmax(model_op)[:, 1]
        source_res["confidence"].append(soft_pred.item())
        source_res["id"].append(id.item())
        source_res["slice_num"].append(slice_num.item())
    source_res["label"] = [0] * len(source_res["id"])

    for image, _, id, slice_num in tqdm(target_loader, desc='running source loader', position=0, leave=True):
        _, _, bottleneck = model(image.to(args.gpu))
        model_op = classifier(bottleneck.to(args.gpu))
        soft_pred = F.softmax(model_op)[:, 0]
        target_res["confidence"].append(soft_pred.item())
        target_res["id"].append(id.item())
        target_res["slice_num"].append(slice_num.item())
    target_res["label"] = [1] * len(target_res["id"])
    final_df = pd.concat([pd.DataFrame.from_dict(source_res).sort_values("confidence"),pd.DataFrame.from_dict(target_res).sort_values("confidence", ascending=False)])
    save_df(final_df, args)

def save_df(df: pd.DataFrame, args):
    import os
    algo_path = f"{os.getcwd()}/dataset/{args.algo}/"
    if not os.path.exists(algo_path):
            os.mkdir(algo_path)
    save_path = f"{algo_path}{args.source}_{args.target}.csv"
    df.to_csv(save_path, index=False)
    print(f"saved dataframe to {save_path}")