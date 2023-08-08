"""Training Code."""

import time
import torch
import torch.nn as nn
from torch.nn.functional import softmax
from ema import EMA
from utils import AverageMeter
from optim import get_optimizer
from network import get_network
from data import get_dataloaders
from collections import defaultdict
from evaluate import Metric, evaluate_step


def train_step(model,
               ema,
               X,
               U,
               device,
               amp_flag,
               criterion,
               threshold,
               optimizer,
               lu_weight,
               scaler,
               scheduler):
    """Train single epoch."""
    global global_step

    logs = defaultdict(AverageMeter)
    metric = Metric()

    model.train()
    for sample_x, sample_u in zip(X, U):
        with torch.autocast(device_type='cuda',
                            dtype=torch.float16,
                            enabled=amp_flag):

            # Augmented Datas with different policies (weak and strong)
            (xw, _), y = sample_x
            (uw, us), _ = sample_u

            inputs = torch.cat([xw, uw, us], dim=0)
            outputs = model(inputs.to(device))

            xw_pred, uw_pred, us_pred = torch.split(outputs,
                                                    [xw.shape[0],
                                                     uw.shape[0],
                                                     us.shape[0]])

            # supervised loss
            ls = criterion(xw_pred, y.to(device)).mean()
            total_loss = ls

            # calcuate a indicator
            with torch.no_grad():
                uw_prob = softmax(uw_pred.detach(), dim=1)
                max_prob, hard_label = torch.max(uw_prob, dim=1)
                indicator = max_prob > threshold

            # unsupervised loss
            lu = (criterion(us_pred, hard_label) * indicator).mean()
            total_loss += lu * lu_weight

        # optimization
        optimizer.zero_grad()
        if amp_flag:
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            optimizer.step()

        scheduler.step()
        ema.update()

        global_step += 1
        metric.update_prediction(xw_pred, y)
        logs['Ls'].update(ls.item())
        logs['Mask'].update(torch.mean(indicator.float()).item())
        if lu_weight > 0 and indicator.any():
            logs['Lu'].update(lu.item())

    Acc = metric.calc_accuracy()
    Ls = logs['Ls'].avg
    Lu = logs['Lu'].avg
    Mask = logs['Mask'].avg

    return Acc, Ls, Lu, Mask


def train_network(args):
    """Train a network."""
    global global_step
    global_step = 0
    if args.wandb:
        import wandb

    device = torch.device('cuda')
    if args.amp:
        scaler = torch.cuda.amp.GradScaler()

    # model
    model = get_network(args.network, args.num_classes)
    if args.mode == 'resume':
        ckpt = torch.load(args.load_path, map_location='cpu')
        model.load_state_dict(ckpt['state_dict'])
        ema = EMA(model=model, decay=args.ema_decay, device=device)
        ema.shadow.load_state_dict(ckpt['ema'])
        start_iter = ckpt['iteration']
    else:
        ema = EMA(model=model, decay=args.ema_decay, device=device)
        start_iter = 0
    model.to(device)

    # criterion
    criterion = nn.CrossEntropyLoss(reduction='none')

    # optimizer
    optimizer, scheduler = get_optimizer(model=model,
                                         lr=args.lr,
                                         momentum=args.momentum,
                                         nesterov=args.nesterov,
                                         weight_decay=args.weight_decay,
                                         iterations=args.iterations)
    if args.mode == 'resume':
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])

    # labeled, unlabeled and test data
    X, U, T = get_dataloaders(data=args.data,
                              num_X=args.num_X,
                              include_x_in_u=args.include_x_in_u,
                              augs=args.augs,
                              batch_size=args.batch_size,
                              mu=args.mu)

    print("#"*20 + f"\n{'Start training...':^20s}\n" + "#"*20)
    n_iter = 1024
    for epoch in range(start_iter//n_iter, n_iter):
        train_results = train_step(model=model,
                                   ema=ema,
                                   X=X,
                                   U=U,
                                   device=device,
                                   criterion=criterion,
                                   amp_flag=args.amp,
                                   lu_weight=args.lu_weight,
                                   threshold=args.threshold,
                                   optimizer=optimizer,
                                   scaler=scaler,
                                   scheduler=scheduler)
        Acc, Ls, Lu, Mask = train_results
        test_Acc = evaluate_step(ema.shadow, T, device)

        print((f"{time.ctime()}: "
               f"Iteration: [{global_step}/{args.iterations}], "
               f"Ls: {Ls:1.4f}, Lu: {Lu:1.4f}, Mask: {Mask:1.4f}, "
               f"Accuracy(train/test): [{Acc:1.4f}/{test_Acc:1.4f}]"))

        check_point = {
                'state_dict': model.state_dict(),
                'ema': ema.shadow.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'args': args,
                'iteration': global_step
                }
        torch.save(check_point, args.save_path / 'ckpt.pth')
        if epoch % 10 == 0:
            torch.save(check_point, args.save_path / f"ckpt_{global_step}.pth")

        if args.wandb:
            wandb.log(data={'Ls': Ls,
                            'Lu': Lu,
                            'Train Acc': Acc,
                            'Mask': Mask,
                            'Test Acc': test_Acc},
                      step=global_step)
