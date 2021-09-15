import json
import os
import pickle as pkl 
import argparse
import torch
from torch import nn
import numpy as np
from fedml_api.data_processing.TILES.data_loader import load_data_tiles
from fedml_api.model.TILES.baseline_models import OneDCnnLstm
from sklearn.metrics import classification_report

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=10)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
    if torch.cuda.is_available(): print('GPU available, use GPU')

    batch_size = 10
    data = load_data_tiles(batch_size=batch_size)

    model = OneDCnnLstm(input_channel=1, num_pred=4)
    restore_path = os.path.join(args.output_dir, 'best_model.pth')
    model.load_state_dict(torch.load(restore_path))
    model.to(device)
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)

    pred_out = []
    target_out = []
    metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }
    with torch.no_grad():
        for batch_idx, (x, target) in enumerate(data):
            x = x.to(device)
            target = target.to(device)
            pred = model(x)
            loss = criterion(pred, target)

            _, predicted = torch.max(pred, -1)
            correct = predicted.eq(target).sum()
            pred_out.extend(predicted.detach().cpu().numpy())
            target_out.extend(target.detach().cpu().numpy())
            metrics['test_correct'] += correct.item()
            metrics['test_loss'] += loss.item() * target.size(0)
            metrics['test_total'] += target.size(0)
        r = classification_report(np.array(target_out).reshape(-1), np.array(pred_out).reshape(-1), \
            output_dict=True, target_names=['other', 'go_to_bed', 'sleep', 'wake_up'])
        metrics['f1score'] = r['weighted avg']['f1-score']
        print(metrics)

        f = open(os.path.join(args.output_dir, 'evaluate.json'), 'w')
        json.dump(r, f)
        f.close()



