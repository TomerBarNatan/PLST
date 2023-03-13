from utils import load_model, get_batch
from models.classifier import FCDiscriminator
from models.unet import UNet2D
import torch
import torch.optim as optim
from tqdm import tqdm
from torch import nn
from torch.utils import data
import torch.nn.functional as F

best_metric = -1
low_source_metric = 1.1

prev_d_score = 0


def site_classification(model_path, train_loader, target_loader, test_ds, val_ds_source, args,
                        config, return_classifier = False, msm=False):
    print(f"site classification\nmodel path is {model_path}")
    best = 0
    model = UNet2D(config.n_channels, n_chans_out=config.n_chans_out, get_bottleneck=True)
    model = load_model(model, model_path, True)
    model.to(args.gpu)
    if config.parallel_model:
        model = torch.nn.DataParallel(model, device_ids=[6,7])
    classifier = FCDiscriminator()
    classifier.train()
    classifier.to(args.gpu)
    if config.parallel_model:
        classifier = torch.nn.DataParallel(classifier, device_ids=[6,7])
    criterion = nn.CrossEntropyLoss()
    if config.parallel_model:
        classifier = torch.nn.DataParallel(classifier, device_ids=[6,7])

    train_loader.dataset.yield_id = True
    target_loader.dataset.yield_id = True
    train_loader_iter = iter(train_loader)
    target_loader_iter = iter(target_loader)
    optimizer = optim.Adam(classifier.parameters(),
                           lr=1e-3, weight_decay=args.weight_decay)
    epochs = 600
    for i in tqdm(range(epochs), "iteration", position=0, leave=True):
        optimizer.zero_grad()
        source_train_batch = get_batch(train_loader, train_loader_iter)
        target_batch = get_batch(target_loader, target_loader_iter)

        source_train_images, _, _, _ = source_train_batch
        target_train_images, _, _, _ = target_batch

        _, _, bottleneck_source = model(source_train_images.to(args.gpu))
        _, _, bottleneck_target = model(target_train_images.to(args.gpu))


        source_preds = classifier(bottleneck_source.to(args.gpu))
        target_preds = classifier(bottleneck_target.to(args.gpu))
        labels = torch.concat((torch.zeros(source_preds.shape[0]), torch.ones(target_preds.shape[0]))).type(torch.LongTensor).to(args.gpu)
        loss = criterion(torch.concat((source_preds, target_preds)), labels)
        loss.backward()
        optimizer.step()
        del bottleneck_source
        del bottleneck_target
        del source_preds
        del target_preds
        if i % 100 == 0 or i == epochs - 1:
            res = statistics(model, classifier,val_ds_source, test_ds, args)
            if res > best:
                best = res
                print("new best classifier saved")
            print(f"res is {res}")
    if return_classifier:
        return classifier, model

def statistics(model, classifier, source_loader, target_loader, args):

    total = 0
    loader = data.DataLoader(target_loader, batch_size=1, shuffle=False)
    correct = 0
    for images, _, _, _ in tqdm(loader, desc='running test loader', position=0, leave=True):
        total += images.shape[0]
        _, _, bottleneck = model(images.to(args.gpu))
        model_op = classifier(bottleneck.to(args.gpu))
        soft_pred = F.softmax(model_op)
        preds = torch.argmax(soft_pred, dim=1)
        correct += torch.count_nonzero(preds)


    loader = data.DataLoader(source_loader, batch_size=1, shuffle=False)
    for images, _, _, _ in tqdm(loader, desc='running test loader', position=0, leave=True):
        total += images.shape[0]
        _, _, bottleneck = model(images.to(args.gpu))
        model_op = classifier(bottleneck.to(args.gpu))
        soft_pred = F.softmax(model_op)
        preds = torch.argmax(soft_pred, dim=1)
        correct += 1 - torch.count_nonzero(preds)
    return correct / total