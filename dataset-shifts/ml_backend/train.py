from __future__ import print_function
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score
from utils import get_scores, test_auc
from itertools import cycle

def classifier(model, optimizer, train_dataloader, **kwargs):
    criterion = nn.BCELoss() 
    device = next(model.parameters()).device
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets, _) in enumerate(tqdm(train_dataloader, desc="training")):
        inputs, targets = inputs.to(device), targets.to(device).type(torch.float)
        optimizer.zero_grad()

        outputs = model(inputs)
        outputs = torch.sigmoid(model(inputs))
        loss = criterion(outputs, targets.unsqueeze(1)).mean()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        correct  += (outputs.squeeze().round() == targets).sum().item()
        total += targets.size(0)

    print(correct / float(total))

def ae(model, optim, train_dataloader, **kwargs):
    train_criteria = torch.nn.MSELoss()
    device = next(model.parameters()).device
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    t = tqdm(train_dataloader, desc="training")
    for batch_idx, (inputs, targets, _) in enumerate(t):
        inputs, targets = inputs.to(device), targets.to(device).type(torch.float)
        optim.zero_grad()

        x_hat = torch.tanh(model(inputs))

        loss = ((x_hat - inputs)**2).mean()
        loss.backward()
        optim.step()
        
        train_loss += loss.item()
        t.set_description(f"{loss.item():.4f}")

    print(train_loss)

def vae_loss_function(recon_x, x, mu, logvar):
    BCE = torch.mean((recon_x - x)**2)
    KLD = -0.5 * torch.mean(torch.mean(1 + logvar - mu.pow(2) - logvar.exp(), 1))
    return BCE, KLD * 0.1

def vae(model, optim, train_dataloader, **kwargs):
    train_criteria = torch.nn.MSELoss()
    device = next(model.parameters()).device
    model.train()

    train_loss = 0
    correct = 0
    total = 0
    t = tqdm(train_dataloader, desc="training")
    for batch_idx, (inputs, targets, _) in enumerate(t):
        inputs = inputs.to(device)
        optim.zero_grad()

        rec, mu, logvar = model(inputs)

        loss_re, loss_kl = vae_loss_function(rec, inputs, mu, logvar)
        loss = loss_re + loss_kl
        loss.backward()
        optim.step()
        
        train_loss += loss.item()
        t.set_description(f"{loss.item():.4f}")

    print(train_loss)

def dre(SugiyamaNet, size, train_loader, test_loader, inlier_data, outlier_data, epochs, device):

    model = SugiyamaNet(size).to(device)
    retmodel = SugiyamaNet(size).to(device)

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)

    best_auc = 0
    
    for e in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (batch) in enumerate(train_loader):
            train_ex = batch.to(device) 
            test_ex = next(iter(test_loader))[:train_ex.size(0)].to(device)

            optimizer.zero_grad()
            out_train = model(train_ex)
            out_test  = model(test_ex)
            loss = (out_test - torch.log(out_train)).mean()
            total_loss+=loss.item()

            loss.backward()
            optimizer.step()

        known_score   = get_scores(model, inlier_data )
        unknown_score = get_scores(model, outlier_data)
        cur_auc = test_auc(known_score, unknown_score)
        if cur_auc > best_auc:
            best_auc = cur_auc
            retmodel.load_state_dict(model.state_dict())

        print(f'Total loss:{total_loss}')
    print("best auc ", best_auc)
    return retmodel

def combo_dre(model, size, train_loader, test_loader, inlier_data, outlier_data, epochs, device):
    model.train()
    for param in model.parameters():
        param.requires_grad = True
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    


    best_auc = 0
    
    for e in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader):
            train_ex = batch[0].to(device) 
            test_ex = next(iter(test_loader))[0][:train_ex.size(0)].to(device)

            optimizer.zero_grad()
            out_train = model(train_ex)
            out_test  = model(test_ex)
            loss = (out_test - torch.log(out_train)).mean()
            total_loss+=loss.item()

            loss.backward()
            optimizer.step()

        known_score   = get_scores(model, inlier_data )
        unknown_score = get_scores(model, outlier_data)
        cur_auc = test_auc(known_score, unknown_score)
        if cur_auc > best_auc:
            best_auc = cur_auc

        print(f'Total loss:{total_loss}')

    print("best auc ", best_auc)
    print("cur auc ", cur_auc)
    return model

eps = 1e-6
def dre_scratch(model, optimizer, train_dataloader, test_dataloader, **kwargs):
    device = next(model.parameters()).device
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets, _) in enumerate(tqdm(train_dataloader, desc="training")):
        train_ex = inputs.to(device)
        test_ex = next(iter(test_dataloader))[0][:train_ex.size(0)].to(device)

        optimizer.zero_grad()
        out_train = F.softplus(eps+ model(train_ex))
        out_test  = F.softplus(eps+ model(test_ex))
        loss = (out_test - torch.log(out_train)).mean()

        optimizer.step()

        train_loss += loss.item()


    print(train_loss)

def imagenet(**kwargs):
    exit("cant train imagenet, only use a pretrained model")

def gan():
    for epoch in range(nb_epochs):
        print('EPOCH {:d} / {:d}'.format(epoch + 1, nb_epochs))
        G_losses, D_losses = utils.AvgMeter(), utils.AvgMeter()
        start_epoch = datetime.datetime.now()

        avg_time_per_batch = utils.AvgMeter()
        # Mini-batch SGD
        for batch_idx, (x, _) in enumerate(data_loader):

            # Critic update ratio
            if self.gan_type == 'wgan':
                n_critic = 20 if g_iter < 50 or (g_iter + 1) % 500 == 0 else self.n_critic
            else:
                n_critic = self.n_critic

            # Training mode
            self.gan.G.train()

            # Discard last examples to simplify code
            if x.size(0) != self.batch_size:
                break
            batch_start = datetime.datetime.now()

            # Print progress bar
            utils.progress_bar(batch_idx, self.batch_report_interval,
                G_losses.avg, D_losses.avg)

            x = Variable(x)
            if torch.cuda.is_available() and self.use_cuda:
                x = x.cuda()

            # Update discriminator
            D_loss, fake_imgs = self.gan.train_D(x, self.D_optimizer, self.batch_size)
            D_losses.update(D_loss, self.batch_size)
            d_iter += 1

            # Update generator
            if batch_idx % n_critic == 0:
                G_loss = self.gan.train_G(self.G_optimizer, self.batch_size)
                G_losses.update(G_loss, self.batch_size)
                g_iter += 1

            batch_end = datetime.datetime.now()
            batch_time = int((batch_end - batch_start).total_seconds() * 1000)
            avg_time_per_batch.update(batch_time)

            # Report model statistics
            if (batch_idx % self.batch_report_interval == 0 and batch_idx) or \
                self.batch_report_interval == self.num_batches:
                G_all_losses.append(G_losses.avg)
                D_all_losses.append(D_losses.avg)
                utils.show_learning_stats(batch_idx, self.num_batches, G_losses.avg, D_losses.avg, avg_time_per_batch.avg)
                [k.reset() for k in [G_losses, D_losses, avg_time_per_batch]]
                self.eval(100, epoch=epoch, while_training=True)
                # print('Critic iter: {}'.format(g_iter))

            # Save stats
            if batch_idx % self.save_stats_interval == 0 and batch_idx:
                stats = dict(G_loss=G_all_losses, D_loss=D_all_losses)
                self.save_stats(stats)

        # Save model
        utils.clear_line()
        print('Elapsed time for epoch: {}'.format(utils.time_elapsed_since(start_epoch)))
        self.gan.save_model(self.ckpt_path, epoch)
        self.eval(100, epoch=epoch, while_training=True)