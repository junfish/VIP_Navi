import os
import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim import lr_scheduler
from tensorboardX import SummaryWriter
import wandb
from model import model_parser
from model import PoseLoss
from pose_utils import *
import datetime


class Solver():
    def __init__(self, data_loader, config):
        self.data_loader = data_loader
        self.config = config

        # do not use dropout if not bayesian mode
        # if not self.config.bayesian:
        #     self.config.dropout_rate = 0.0

        self.device = torch.device("cuda:" + self.config.gpu_ids if torch.cuda.is_available() else "cpu")


        self.model = model_parser(self.config.model, self.config.fixed_weight, self.config.dropout_rate,
                                  self.config.bayesian)

        self.criterion = PoseLoss(self.device, self.config.sx, self.config.sq, self.config.learn_beta)

        self.print_network(self.model, self.config.model)
        self.model_name = self.config.model
        self.data_name = self.config.proj_path.split('/')[-1]
        if self.config.mode == 'train':
            self.model_save_path = 'models_%s_%s-%s' % (self.model_name, self.data_name, datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S"))
        elif self.config.mode == 'test':
            self.model_save_path = 'models_%s_%s' % (self.model_name, self.data_name)
        # self.model_save_path = 'models_NCLT_cam4_2seqs_3m'
        self.summary_save_path = 'summary_%s_%s' % (self.model_name, self.data_name)
        self.test_save_path = 'test_%s_%s' % (self.model_name, self.data_name)

        if self.config.pretrained_model:
            self.load_pretrained_model()


        if self.config.sequential_mode:
            self.set_sequential_mode()

    # Inner Functions #
    def set_sequential_mode(self):
        if self.config.sequential_mode == 'model':
            self.model_save_path = 'models/%s/models_%s' % (self.config.sequential_mode, self.config.model)
            self.summary_save_path = 'summaries/%s/summary_%s' % (self.config.sequential_mode, self.config.model)
        elif self.config.sequential_mode == 'fixed_weight':
            self.model_save_path = 'models/%s/models_%d' % (self.config.sequential_mode, int(self.config.fixed_weight))
            self.summary_save_path = 'summaries/%s/summary_%d' % (self.config.sequential_mode, int(self.config.fixed_weight))
        elif self.config.sequential_mode == 'batch_size':
            self.model_save_path = 'models/%s/models_%d' % (self.config.sequential_mode, self.config.batch_size)
            self.summary_save_path = 'summaries/%s/summary_%d' % (self.config.sequential_mode, self.config.batch_size)
        elif self.config.sequential_mode == 'learning_rate':
            self.model_save_path = 'models/%s/models_%f' % (self.config.sequential_mode, self.config.lr)
            self.summary_save_path = 'summaries/%s/summary_%f' % (self.config.sequential_mode, self.config.lr)
        elif self.config.sequential_mode == 'beta':
            self.model_save_path = 'models/%s/models_%d' % (self.config.sequential_mode, self.config.beta)
            self.summary_save_path = 'summaries/%s/summary_%d' % (self.config.sequential_mode, self.config.beta)
        else:
            assert 'Unvalid sequential mode'

    def load_pretrained_model(self):
        model_path = self.model_save_path + '/%s_net.pth' % self.config.pretrained_model
        self.model.load_state_dict(torch.load(model_path))
        print('Load pretrained network: ', model_path)

    def print_network(self, model, name):
        num_params = 0
        for param in model.parameters():
            num_params += param.numel()

        print('*' * 20)
        print(name)
        print(model)
        print('*' * 20)

    def loss_func(self, input, target):
        diff = torch.norm(input-target, dim=1)
        diff = torch.mean(diff)
        return diff

    def train(self):

        self.model = self.model.to(self.device)
        self.criterion = self.criterion.to(self.device)
        if self.config.learn_beta:
            print("Beta is also trainable in this running!")
            print("********************")
            optimizer = optim.Adam([{'params': self.model.parameters()},
                                    {'params': [self.criterion.sx, self.criterion.sq]}],
                                   lr = self.config.lr,
                                   weight_decay =0.0005)
        else:
            optimizer = optim.Adam(self.model.parameters(),
                                   lr=self.config.lr,
                                   weight_decay = 0.0005)


        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.config.num_epochs_decay, gamma=0.1)

        num_epochs = self.config.num_epochs

        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # Setup for tensorboard
        use_tensorboard = self.config.use_tensorboard
        if use_tensorboard:
            if not os.path.exists(self.summary_save_path):
                os.makedirs(self.summary_save_path)
            writer = SummaryWriter(log_dir=self.summary_save_path)
        else:
            wandb.init(
                project="HST_Navi",
                name='run_%s_%s-%s' % (self.model_name, self.data_name, datetime.datetime.now().strftime("%Y-%m-%d/%H:%M:%S")),
                entity='juy220',
                # config=self.config
            )

        since = time.time()
        n_iter = 0

        # For pretrained network
        start_epoch = 0
        if self.config.pretrained_model:
            start_epoch = int(self.config.pretrained_model)

        # Pre-define variables to get the best model
        best_train_loss = 10000
        best_val_loss = 10000
        best_train_model = None
        best_val_model = None


        for epoch in range(start_epoch, num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs-1))
            print('-'*20)

            error_train = []
            error_pos_train = []
            error_ori_train = []
            error_val = []
            error_pos_val =[]
            error_ori_val = []

            for phase in ['train', 'val']:

                if phase == 'train':
                    # move it after optimizer.step()
                    self.model.train()
                else:
                    self.model.eval()

                data_loader = self.data_loader[phase]

                for i, (inputs, poses) in enumerate(data_loader):

                    inputs = inputs.to(self.device)
                    poses = poses.to(self.device)

                    # Zero the parameter gradient
                    optimizer.zero_grad()

                    # forward
                    pos_out, ori_out, _ = self.model(inputs)

                    # pos_true = poses[:, :3]
                    # ori_true = poses[:, 3:]
                    pos_true = poses[:, 4:]
                    ori_true = poses[:, :4]

                    ori_out = F.normalize(ori_out, p=2, dim=1)
                    ori_true = F.normalize(ori_true, p=2, dim=1)

                    loss, _, _ = self.criterion(pos_out, ori_out, pos_true, ori_true)
                    loss_print = self.criterion.loss_print[0]
                    loss_pos_print = self.criterion.loss_print[1]
                    loss_ori_print = self.criterion.loss_print[2]

                    # loss_pos = F.mse_loss(pos_out, pos_true)
                    # loss_ori = F.mse_loss(ori_out, ori_true)

                    # loss_pos = F.l1_loss(pos_out, pos_true)
                    # loss_ori = F.l1_loss(ori_out, ori_true)
                    # loss_pos = self.loss_func(pos_out, pos_true)
                    # loss_ori = self.loss_func(ori_out, ori_true)

                    # loss = loss_pos + beta * loss_ori

                    # loss_print = loss.item()
                    # loss_ori_print = loss_ori.item()
                    # loss_pos_print = loss_pos.item()

                    if use_tensorboard:
                        if phase == 'train':
                            error_train.append(loss_print)
                            error_pos_train.append(loss_pos_print)
                            error_ori_train.append(loss_ori_print)
                            writer.add_scalar('loss/overall_loss', loss_print, n_iter)
                            writer.add_scalar('loss/position_loss', loss_pos_print, n_iter)
                            writer.add_scalar('loss/rotation_loss', loss_ori_print, n_iter)
                            if self.config.learn_beta:
                                writer.add_scalar('param/sx', self.criterion.sx.item(), n_iter)
                                writer.add_scalar('param/sq', self.criterion.sq.item(), n_iter)

                        elif phase == 'val':
                            error_val.append(loss_print)
                            error_pos_val.append(loss_pos_print)
                            error_ori_val.append(loss_ori_print)
                    else:
                        if phase == 'train':
                            error_train.append(loss_print)
                            error_pos_train.append(loss_pos_print)
                            error_ori_train.append(loss_ori_print)
                            wandb.log({
                                'loss/overall_loss': loss_print,
                                'step': n_iter
                            })
                            wandb.log({
                                'loss/position_loss': loss_pos_print,
                                'step': n_iter
                            })
                            wandb.log({
                                'loss/rotation_loss': loss_ori_print,
                                'step': n_iter
                            })

                            if self.config.learn_beta:
                                wandb.log({
                                    'param/sx': self.criterion.sx.item(),
                                    'step': n_iter
                                })
                                wandb.log({
                                    'param/sq': self.criterion.sq.item(),
                                    'step': n_iter
                                })
                        elif phase == 'val':
                            error_val.append(loss_print)
                            error_pos_val.append(loss_pos_print)
                            error_ori_val.append(loss_ori_print)


                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        n_iter += 1
                if phase == 'train':
                    scheduler.step()

                    # print('{}th {} Loss: total loss {:.3f} / pos loss {:.3f} / ori loss {:.3f}'.format(i, phase, loss_print, loss_pos_print, loss_ori_print))

            # For each epoch
            error_train_mean = sum(error_train) / len(error_train)
            error_train_pos_mean = sum(error_pos_train) / len(error_pos_train)
            error_train_ori_mean = sum(error_ori_train) / len(error_ori_train)

            error_val_mean = sum(error_val) / len(error_val)
            error_val_pos_mean = sum(error_pos_val) / len(error_pos_val)
            error_val_ori_mean = sum(error_ori_val) / len(error_ori_val)


            error_train_median = np.median(error_train)
            error_train_pos_median = np.median(error_pos_train)
            error_train_ori_median = np.median(error_ori_train)

            error_val_median = np.median(error_val)
            error_val_pos_median = np.median(error_pos_val)
            error_val_ori_median = np.median(error_ori_val)

            if (epoch+1) % self.config.model_save_step == 0:
                save_filename = self.model_save_path + '/%s_net.pth' % epoch
                # save_path = os.path.join('models', save_filename)
                torch.save(self.model.cpu().state_dict(), save_filename)
                if torch.cuda.is_available():
                    self.model.to(self.device)

            if error_train_median < best_train_loss:
                best_train_loss = error_train_median
                best_train_model = epoch
            if error_val_median < best_val_loss:
                best_val_loss = error_val_median
                best_val_model = epoch
                save_filename = self.model_save_path + '/best_net.pth'
                torch.save(self.model.cpu().state_dict(), save_filename)
                if torch.cuda.is_available():
                    self.model.to(self.device)

            print('Train and Validaion error (median): {} / {}'.format(error_train_median, error_val_median))
            print('=' * 40)
            print('=' * 40)

            if use_tensorboard:
                writer.add_scalars('loss/train_val (median)', {'total_train':error_train_median, 'pos_train': error_train_pos_median, 'ori_train': error_train_ori_median, 'total_val':error_val_median, 'pos_val': error_val_pos_median, 'ori_val': error_val_ori_median}, epoch)
                writer.add_scalars('loss/train_val (mean)', {'total_train': error_train_mean, 'pos_train': error_train_pos_mean, 'ori_train': error_train_ori_mean, 'total_val': error_val_mean, 'pos_val': error_val_pos_mean, 'ori_val': error_val_ori_mean}, epoch)
                writer.add_scalar('learning_rate', scheduler.get_last_lr(), epoch)  # get_lr() vs get_last_lr()
            else:
                # Logging the median train and validation losses along with the epoch
                wandb.log({
                    'median_error/total_train': error_train_median,
                    'median_error/pos_train': error_train_pos_median,
                    'median_error/ori_train': error_train_ori_median,
                    'median_error/total_val': error_val_median,
                    'median_error/pos_val': error_val_pos_median,
                    'median_error/ori_val': error_val_ori_median,
                    'epoch': epoch
                })

                # Logging the mean train and validation losses along with the epoch
                wandb.log({
                    'mean_error/total_train': error_train_mean,
                    'mean_error/pos_train': error_train_pos_mean,
                    'mean_error/ori_train': error_train_ori_mean,
                    'mean_error/total_val': error_val_mean,
                    'mean_error/pos_val': error_val_pos_mean,
                    'mean_error/ori_val': error_val_ori_mean,
                    'epoch': epoch
                })

                # Logging the learning rate along with the epoch
                # Note: Ensure scheduler.get_last_lr() gives the expected format. If it returns a list, you might want to access the first element using [0].
                learning_rate = scheduler.get_last_lr()[0] if isinstance(scheduler.get_last_lr(),
                                                                         list) else scheduler.get_last_lr()
                wandb.log({
                    'learning_rate': learning_rate,
                    'param/sx': self.criterion.sx.item(),
                    'param/sq': self.criterion.sq.item(),
                    'epoch': epoch
                })


        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

        if self.config.sequential_mode:
            f = open(self.summary_save_path + '/train.csv', 'w')

            f.write('{},{}\n{},{}'.format(best_train_loss, best_train_model,
                                          best_val_loss, best_val_model))
            f.close()
            # return (best_train_loss, best_train_model), (best_val_loss, best_val_model)

    def test(self):
        if not os.path.exists(self.test_save_path + '/' + self.config.train_time):
            os.makedirs(self.test_save_path + '/' + self.config.train_time)
        f = open(self.test_save_path + '/' + self.config.train_time + '/test_result.csv', 'w')

        device = torch.device("cuda:" + self.config.gpu_ids if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)
        self.model.eval()


        if self.config.test_model is None:
            test_model_path = self.model_save_path + '-' + self.config.train_time + '/best_net.pth'
        else:
            test_model_path = self.model_save_path + '-' + self.config.train_time + '/{}_net.pth'.format(self.config.test_model)

        print('Load pretrained model: ', test_model_path)
        self.model.load_state_dict(torch.load(test_model_path))

        total_pos_loss = 0
        total_ori_loss = 0
        pos_loss_arr = []
        ori_loss_arr = []
        true_list = []
        estim_list = []
        if self.config.bayesian:
            pred_mean = []
            pred_var = []

        num_data = len(self.data_loader)

        for i, (inputs, poses) in enumerate(self.data_loader):
            print(i)

            inputs = inputs.to(self.device)

            # forward
            if self.config.bayesian:
                num_bayesian_test = 100
                pos_array = torch.Tensor(num_bayesian_test, 3)
                ori_array = torch.Tensor(num_bayesian_test, 4)

                for i in range(num_bayesian_test):
                    pos_single, ori_single, _ = self.model(inputs)
                    pos_array[i, :] = pos_single
                    ori_array[i, :] = F.normalize(ori_single, p=2, dim=1)

                pose_quat = torch.cat((ori_array, pos_array), 1).detach().cpu().numpy()
                pred_mean, pred_var = fit_gaussian(pose_quat)

                pos_var = np.sum(pred_var[4:])
                ori_var = np.sum(pred_var[:4])

                pos_out = pred_mean[4:]
                ori_out = pred_mean[:4]
            else:
                pos_out, ori_out, _ = self.model(inputs)
                pos_out = pos_out.squeeze(0).detach().cpu().numpy()
                ori_out = F.normalize(ori_out, p=2, dim=1)
                ori_out = ori_out.squeeze(0).detach().cpu().numpy()
                # ori_out = quat_to_euler(ori_out)
                # print('pos out', pos_out)
                # print('ori_out', ori_out)
            #
            # pos_true = poses[:, :3].squeeze(0).numpy()
            # ori_true = poses[:, 3:].squeeze(0).numpy()

            pos_true = poses[:, 4:].squeeze(0).numpy()
            ori_true = poses[:, :4].squeeze(0).numpy()

            # ori_true = quat_to_euler(ori_true)
            # print('pos true', pos_true)
            # print('ori true', ori_true)
            loss_pos_print = array_dist(pos_out, pos_true)
            # loss_ori_print = array_dist(ori_out, ori_true)
            loss_ori_print = quat_dist(ori_out, ori_true)

            true_list.append(np.hstack((pos_true, ori_true)))
            
            # if loss_pos_print < 20:
            #     estim_list.append(np.hstack((pos_out, ori_out)))
            estim_list.append(np.hstack((pos_out, ori_out)))

            # ori_out = F.normalize(ori_out, p=2, dim=1)
            # ori_true = F.normalize(ori_true, p=2, dim=1)
            #
            # loss_pos_print = F.pairwise_distance(pos_out, pos_true, p=2).item()
            # loss_ori_print = F.pairwise_distance(ori_out, ori_true, p=2).item()

            # loss_pos_print = F.l1_loss(pos_out, pos_true).item()
            # loss_ori_print = F.l1_loss(ori_out, ori_true).item()

            # loss_pos_print = self.loss_func(pos_out, pos_true).item()
            # loss_ori_print = self.loss_func(ori_out, ori_true).item()

            # print(pos_out)
            # print(pos_true)

            total_pos_loss += loss_pos_print
            total_ori_loss += loss_ori_print

            pos_loss_arr.append(loss_pos_print)
            ori_loss_arr.append(loss_ori_print)



            if self.config.bayesian:
                print('{}th Error: pos error {:.3f} / ori error {:.3f}'.format(i, loss_pos_print, loss_ori_print))
                print('{}th std: pos / ori', pos_var, ori_var)
                f.write('{},{},{},{}\n'.format(loss_pos_print, loss_ori_print, pos_var, ori_var))

            else:
                print('{}th Error: pos error {:.3f} / ori error {:.3f}'.format(i, loss_pos_print, loss_ori_print))



        # position_error = sum(pos_loss_arr)/len(pos_loss_arr)
        # rotation_error = sum(ori_loss_arr)/len(ori_loss_arr)
        position_error = np.median(pos_loss_arr)
        rotation_error = np.median(ori_loss_arr)

        print('=' * 20)
        print('Overall median pose errer {:.3f} / {:.3f}'.format(position_error, rotation_error))
        print('Overall average pose errer {:.3f} / {:.3f}'.format(np.mean(pos_loss_arr), np.mean(ori_loss_arr)))
        f.close()

        if self.config.save_result:
            f_true = self.test_save_path + '/' + self.config.train_time + '/true.csv'
            f_estim = self.test_save_path + '/' + self.config.train_time + '/estim.csv'
            f_error = self.test_save_path + '/' + self.config.train_time + '/error.csv'
            np.savetxt(f_true, true_list, delimiter=',')
            np.savetxt(f_estim, estim_list, delimiter=',')
            np.savetxt(f_error, np.hstack((np.array(pos_loss_arr).reshape(-1, 1), np.array(ori_loss_arr).reshape(-1, 1))), delimiter=',')

        if self.config.sequential_mode:
            f = open(self.test_save_path + '/' + self.config.train_time + '/test.csv', 'w')
            f.write('{},{}'.format(position_error, rotation_error))
            f.close()
            # return position_error, rotation_error

