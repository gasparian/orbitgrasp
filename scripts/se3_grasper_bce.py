import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from scripts.se3_grasp_bce import OrbitGrasp
from utils.torch_utils import write_training, write_log
import warnings
from tqdm import tqdm
from utils.FeaturedPoints import FeaturedPoints
from utils.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")


class OrbitGrasper:

    def __init__(self, device, param_dir='./store/random_obs', lr=1e-5, load=0, num_channel=36, lmax=3, mmax=2,
                 load_name=None, training_config=None):
        self.device = device
        self.model = OrbitGrasp(device=self.device, num_channel=num_channel, lr=lr, lmax=lmax, mmax=mmax)

        self.load_name = load_name

        self.training_contain_mask = training_config['training_contain_mask']

        self.scheduler = LRScheduler(self.model.optim, training_config)

        if not os.path.exists(param_dir):
            os.makedirs(param_dir)

        self.parameter_dir = param_dir
        if not os.path.exists(self.parameter_dir):
            os.makedirs(self.parameter_dir)

        if load != 0:
            self.load()
            self.epoch_num = load + 1
            # scheduler step
            for _ in range(load):
                self.scheduler.step()
        else:
            self.epoch_num = 1

    def train_test_save_aug(self, train_dataset, test_dataset=None, tr_epoch=20, verbose=True,
                            save_interval=1, log=True, balance=True):

        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

        for epoch_num in range(self.epoch_num, tr_epoch + 1):
            train_dataset.aug_data_before_epoch()
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)

            train_step = 1
            total_loss = 0.
            tst_step = 0
            pbar = tqdm(train_loader, desc='Actions', leave=True)
            for batch in pbar:
                data, grasp_indices_all, labels_all, spherical_harmonics = batch
                x = data.x.squeeze(0).to(self.device)
                n = data.n.squeeze(0).to(self.device)
                b = data.b.squeeze(0).to(self.device)
                data = FeaturedPoints(x=x, n=n, b=b)

                batch = [data, grasp_indices_all.squeeze(0).to(self.device), labels_all.squeeze(0).to(self.device),
                         spherical_harmonics]
                loss = self.model.train(batch, balance=balance)

                if loss is None:
                    continue
                tst_step += 1
                total_loss += loss
                current_lr = self.model.optim.param_groups[0]['lr']
                if log:
                    write_training(self.parameter_dir, epoch_num, train_step, loss)
                if verbose:
                    pbar.set_postfix(
                        {"Epoch:": f'{epoch_num}/ {tr_epoch}', "Step:": f'{train_step}', "Tr loss:": f'{loss:5f}',
                         "Tr avg loss:": f'{(total_loss / tst_step):5f}', "lr:": f'{current_lr}'})

                self.loss = total_loss / tst_step
                train_step = train_step + 1
            write_training(self.parameter_dir, epoch_num, train_step, total_loss / tst_step)

            if test_loader is not None:

                test_step = 1
                test_total_loss = 0.
                test_total_acc = 0.
                test_max_pred = 0.
                t_step = 0
                pbar = tqdm(test_loader, desc='Actions', leave=True)
                for batch in pbar:
                    data, grasp_indices_all, labels_all, harmonics = batch
                    x = data.x.squeeze(0).to(self.device)
                    n = data.n.squeeze(0).to(self.device)
                    b = data.b.squeeze(0).to(self.device)
                    data = FeaturedPoints(x=x, n=n, b=b)
                    grasp_indices_cuda = []
                    labels_cuda = []
                    for grasp_indices, labels in zip(grasp_indices_all, labels_all):
                        if labels.ndim == 1:
                            labels = labels[None]
                        grasp_indices_cuda.append(grasp_indices.to(self.device))
                        labels_cuda.append(labels.to(self.device))

                    batch = [data, grasp_indices_cuda, labels_cuda, harmonics]
                    test_loss, acc, max_pred = self.model.test(batch)

                    if test_loss is None:
                        continue

                    t_step += 1
                    test_total_loss += test_loss
                    test_total_acc += acc
                    test_max_pred += max_pred
                    if verbose:
                        pbar.set_postfix(
                            {"Epoch:": f'{epoch_num}/ {tr_epoch}',
                             "Tst loss:": f'{test_loss:5f}',
                             "Tst avg loss:": f'{(test_total_loss / t_step):5f}',
                             "Tst avg acc:": f'{(test_total_acc / t_step):5f}',
                             "Tst max pred:": f'{(test_max_pred / t_step):5f}'})

                    self.test_loss = test_total_loss / t_step
                    test_step = test_step + 1

                write_log(self.parameter_dir, test_total_loss / t_step, test_total_acc / t_step, test_max_pred / t_step,
                          scene='test')
            self.scheduler.step()
            if epoch_num % save_interval == 0:
                self.save()
            self.epoch_num += 1

    def predict(self, feature_points_list, grasp_indices_list, harmonics_list):

        score_list, feature_list = self.model.predict(feature_points_list, grasp_indices_list, harmonics_list)

        return score_list, feature_list

    def save(self):
        if not os.path.exists(self.parameter_dir):
            os.makedirs(self.parameter_dir)
        fname1 = 'orbitgrasp-ckpt-%d-%.4f.pt' % (self.epoch_num, self.test_loss)
        if self.training_contain_mask:
            fname1 = fname1.replace('.pt', '-mask.pt')
        else:
            fname1 = fname1.replace('.pt', '-no-mask.pt')

        fname1 = os.path.join(self.parameter_dir, fname1)

        self.model.save(fname1)
        print('save the parameters to' + fname1)

    def load(self):
        fname = self.load_name
        fname = os.path.join(self.parameter_dir, fname)

        self.model.load(fname)
        print('Load the parameters from ' + fname)
