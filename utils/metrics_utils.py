import numpy as np
import surface_distance.metrics as surf_dc
import torch
from scipy import ndimage
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from tqdm import tqdm


def sdice(gt, pred, spacing, tolerance=1):
    surface_distances = surf_dc.compute_surface_distances(gt, pred, spacing)

    return surf_dc.compute_surface_dice_at_tolerance(surface_distances, tolerance), dice(gt, pred)


def _connectivity_region_analysis(mask):
    label_im, nb_labels = ndimage.label(mask)

    sizes = ndimage.sum(mask, label_im, range(nb_labels + 1))

    # plt.imshow(label_im)
    label_im[label_im != np.argmax(sizes)] = 0
    label_im[label_im == np.argmax(sizes)] = 1

    return label_im


def dice(gt, pred):
    if gt.shape != pred.shape:
        gt = gt.squeeze(1)
    g = np.zeros(gt.shape)
    p = np.zeros(pred.shape)
    g[gt == 1] = 1
    p[pred == 1] = 1
    return (2 * np.sum(g * p)) / (np.sum(g) + np.sum(p))


def dice_torch(gt, pred, smooth=0):
    if gt.shape != pred.shape:
        gt = gt.squeeze(1)
    g = torch.zeros(gt.shape)
    p = torch.zeros(pred.shape)
    g[gt == 1] = 1
    p[pred == 1] = 1
    return (2 * torch.sum(g * p) + smooth) / (torch.sum(g) + torch.sum(p) + smooth)


def get_sdice(model, ds, device, config):
    if config.debug:
        return 0.5, 0.6
    loader = data.DataLoader(ds, batch_size=1, shuffle=False)
    model.eval()
    prev_id = None
    all_segs = []
    all_preds = []
    done_ids = set()
    all_sdices = []
    all_dices = []
    with torch.no_grad():

        for images, segs, ids, slc_num in tqdm(loader, desc='running test loader'):
            id1 = int(ids[0])
            _, output = model(images.to(device))
            if output.shape[1] == 2:
                output = output.cpu().data.numpy()
                output = np.asarray(np.argmax(output, axis=1), dtype=np.uint8).astype(bool)
            else:
                assert output.shape[1] == 1
                output = (nn.Sigmoid()(output) > 0.5).squeeze(1).cpu().data.numpy()
            segs = segs.squeeze(1).numpy().astype(bool)
            if prev_id is None:
                prev_id = id1
            if id1 != prev_id:
                assert id1 not in done_ids
                done_ids.add(id1)
                id1_str = str(id1)
                while len(id1_str) < 3:
                    id1_str = '0' + id1_str
                sdice1, dice1 = sdice(np.stack(all_segs), np.stack(all_preds), ds.spacing_loader('CC0' + id1_str))
                all_sdices.append(sdice1)
                all_dices.append(dice1)
                all_preds = []
                all_segs = []
            prev_id = id1
            all_preds.append(output[0])
            all_segs.append(segs[0])

    return float(np.mean(all_dices)), float(np.mean(all_sdices))


def get_dice(model, ds, device, config):
    if config.debug:
        return 0.5, 0.6
    model.eval()
    dices = []
    with torch.no_grad():
        for id1, images in tqdm(ds.patches_Allimages.items(), desc='running val or test loader'):
            segs = ds.patches_Allmasks[id1]
            images = Variable(torch.tensor(images)).to(device)
            _, output = model(images)
            if output.shape[1] == 2:
                output = output.cpu().data.numpy()
                output = np.asarray(np.argmax(output, axis=1), dtype=np.uint8).astype(bool)
            else:
                assert output.shape[1] == 1
                output = (nn.Sigmoid()(output) > 0.5).squeeze(1).cpu().data.numpy()
            output = _connectivity_region_analysis(output)
            dices.append(dice(segs, output))
    return float(np.mean(dices)), 0
