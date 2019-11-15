import time
import torch
from progress.bar import Bar
import numpy as np
import pandas as pd


def train_step(train_loader, model, epoch, optimizer, criterion, args):

    # switch to train mode
    model.train()
    epoch_loss = 0.0
    loss_w =args.loss_w

    iters_per_epoch = len(train_loader)
    bar = Bar('Processing {} Epoch -> {} / {}'.format('train', epoch+1, args.epochs), max=iters_per_epoch)
    bar.check_tty = False

    for step, (imagesA, imagesB, imagesC, labels) in enumerate(train_loader):
        start_time = time.time()

        torch.set_grad_enabled(True)

        imagesA = imagesA.cuda()
        imagesB = imagesB.cuda()
        imagesC = imagesC.cuda()

        labels = labels.cuda()

        out_A, out_B, out_C, out_F, combine = model(imagesA, imagesB, imagesC)

        loss_x = criterion(out_A, labels)
        loss_y = criterion(out_B, labels)
        loss_z = criterion(out_C, labels)
        loss_c = criterion(out_F, labels)
        loss_f = criterion(combine, labels)

        lossValue = loss_w[0]*loss_x+loss_w[1]*loss_y+loss_w[2]*loss_z+loss_w[3]*loss_c+loss_w[4]*loss_f


        optimizer.zero_grad()
        lossValue.backward()
        optimizer.step()

        # measure elapsed time
        epoch_loss += lossValue.item()
        end_time = time.time()
        batch_time = end_time - start_time
        # plot progress
        bar_str = '{} / {} | Time: {batch_time:.2f} mins | Loss: {loss:.4f} '
        bar.suffix = bar_str.format(step+1, iters_per_epoch, batch_time=batch_time*(iters_per_epoch-step)/60,
                                    loss=lossValue.item())
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch

    bar.finish()
    return epoch_loss


def validation_step(val_loader, model, criterion):

    # switch to train mode
    model.eval()
    epoch_loss = 0
    iters_per_epoch = len(val_loader)
    bar = Bar('Processing {}'.format('validation'), max=iters_per_epoch)

    for step, (imagesA, imagesB, imagesC, labels) in enumerate(val_loader):
        start_time = time.time()

        imagesA = imagesA.cuda()
        imagesB = imagesB.cuda()
        imagesC = imagesC.cuda()
        labels = labels.cuda()

        _, _, _, _, outputs = model(imagesA, imagesB, imagesC)
        with torch.no_grad():
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

        end_time = time.time()

        # measure elapsed time
        batch_time = end_time - start_time
        bar_str = '{} / {} | Time: {batch_time:.2f} mins'
        bar.suffix = bar_str.format(step + 1, len(val_loader), batch_time=batch_time * (iters_per_epoch - step) / 60)
        bar.next()

    epoch_loss = epoch_loss / iters_per_epoch
    bar.finish()
    return epoch_loss


def save_output(label_test_file, dataPRED, args, save_file):
    label_list = args.label_idx
    n_class = len(label_list)
    datanpPRED = np.squeeze(dataPRED.cpu().numpy())
    df_tmp = pd.read_csv(label_test_file)
    image_names = df_tmp["image"].tolist()

    result = {label_list[i]: datanpPRED[:, i] for i in range(n_class)}
    result['image_name'] = image_names
    out_df = pd.DataFrame(result)

    name_older = ['image_name']
    for i in range(n_class):
        name_older.append(label_list[i])
    out_df.to_csv(save_file, columns=name_older)


