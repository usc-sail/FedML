import logging
from sklearn.metrics import classification_report
import torch
from torch import nn
import os
import json
import numpy as np
try:
    from fedml_core.trainer.model_trainer import ModelTrainer
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer


class MyModelTrainer(ModelTrainer):
    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        model = self.model

        model.to(device)
        model.train()

        # train and update
        criterion = nn.CrossEntropyLoss().to(device)
        if args.client_optimizer == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr)
        else:
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=args.lr,
                                         weight_decay=args.wd, amsgrad=True)

        epoch_loss = []
        for epoch in range(args.epochs):
            batch_loss = []
            for batch_idx, (x, labels) in enumerate(train_data):
                x, labels = x.to(device), labels.to(device)
                model.zero_grad()
                log_probs = model(x)
                loss = criterion(log_probs, labels)
                loss.backward()

                # Uncommet this following line to avoid nan loss
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()
                logging.info('Update Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, (batch_idx + 1) * args.batch_size, len(train_data) * args.batch_size,
                           100. * (batch_idx + 1) / len(train_data), loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            logging.info('Client Index = {}\tEpoch: {}\tLoss: {:.6f}'.format(
                self.id, epoch, sum(epoch_loss) / len(epoch_loss)))

    def test(self, test_data, device, args, save_model=False):
        model = self.model

        model.to(device)
        model.eval()
        pred_out = []
        target_out = []
        metrics = {
            'test_correct': 0,
            'test_loss': 0,
            'test_total': 0
        }

        criterion = nn.CrossEntropyLoss().to(device)

        with torch.no_grad():
            for batch_idx, (x, target) in enumerate(test_data):
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
            # logging.info('pred: {}'.format(pred_out))
            # logging.info('target: {}'.format(target_out))
            r = classification_report(np.array(target_out).reshape(-1), np.array(pred_out).reshape(-1), \
                output_dict=True, target_names=['other', 'go_to_bed', 'sleep', 'wake_up'])
            metrics['f1score'] = r['weighted avg']['f1-score']
            if save_model:
                os.makedirs(args.output_dir, exist_ok=True)
                best_metrics_file = os.path.join(args.output_dir, 'best_metrics.json')
                best_model_path = os.path.join(args.output_dir, 'best_model.pth')
                if os.path.isfile(best_metrics_file):
                    best_metrics = json.load(open(best_metrics_file, 'r'))
                    best_f1score = float(best_metrics['weighted avg']['f1-score'])
                    if best_f1score <= metrics['f1score']:
                        f = open(best_metrics_file, 'w')
                        json.dump(r, f)
                        f.close()
                        torch.save(model.state_dict(), best_model_path)
                        logging.info('saved model with f1-score: {} at {}'.format(metrics['f1score'], best_model_path))
                else:
                    f = open(best_metrics_file, 'w')
                    json.dump(r, f)
                    f.close()
                    torch.save(model.state_dict(), best_model_path)
                    logging.info('saved model with f1-score: {} at {}'.format(metrics['f1score'], best_model_path))
        return metrics

    def test_on_the_server(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        return False
