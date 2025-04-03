import os
import numpy as np
import torch
import torch.optim as optim
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from ffdnet.utils.data_utils import batch_psnr, svd_orthogonalization, init_logger
from ffdnet.utils.train_utils import load_dataset_and_dataloader, create_model, \
			init_loss, resume_training, create_input_variables, compute_loss, get_lr

# try to import the autocast command
try:
    with autocast(device_type='cuda'):
        pred = model(imgn, stdn_var)
        loss = compute_loss(criterion, pred, noise, imgn).cpu()
except NameError:
    pred = model(imgn, stdn_var)
    loss = compute_loss(criterion, pred, noise, imgn).cpu()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train(args):
  r"""Main training function.

  Args:
    gray: train grayscale image denoising instead of RGB 
    experiment_name: path of log directory inside of the folder experiments 
    batch_size: training batch size 
    val_batch_size: validation batch size 
    epochs: number of total training epochs 
    resume_training: resume training from a previous checkpoint 
    lr: Initial learning rate
    no_orthog: don't perform orthogonalization as regularization 
    save_every: number of training steps to log psnr and perform
    save_every_epochs: number of training epochs to save state 
    noiseIntL: noise training interval 
    val_noiseL: noise level used on validation set 
    filter: apply wiener filter or default (AWGN artifical noise) to extract noise from dataset 
    traindbf: h5py file containing the images for training the net 
    valdbf: h5py file containing the images for validating the net 
    gpu_fraction: if it is specified, it tries to set the upperbound GPU limit (only if set_process_memory_fraction is implemented), otherwise it only notifies the user
  """
  # Import và sử dụng compatibility layer
  from ffdnet.compatibility import set_memory_allocation
  set_memory_allocation(args.gpu_fraction)

	# Performs the main training loop

  # Load dataset
  datasets, dataloaders = load_dataset_and_dataloader(args)

	# Init loggers
  (args.experiment_name).mkdir(parents=True, exist_ok=True)
  writer = SummaryWriter(args.experiment_name)
  logger = init_logger(args)

  # Create model
  model = create_model(args)

  # Define loss
  criterion = init_loss()
  # Optimizer
  optimizer = optim.Adam(model.parameters(), lr=args.lr)
  # Scheduler
  scheduler = ReduceLROnPlateau(optimizer, 'min', patience=40)

  # Resume training or start anew
  training_params, val_params, start_epoch, args = resume_training(args, model, optimizer)
  scheduler.num_bad_epochs = training_params['num_bad_epochs']

	# Training
  for epoch in range(start_epoch, args.epochs):
    print('learning rate %f' % args.lr)

    # Set up the metrics
    metrics = {'loss': {'train': 0, 'val': 0},
               'psnr': {'train': 0, 'val': 0}}

    for phase in ['train', 'val']:      
      total_batch_number = len(dataloaders[phase])
      pbar = tqdm(total=total_batch_number)
      
      # We initialize the batch loss and the epoch loss
      running_loss = 0.0
      running_psnr = 0.0
      epoch_loss = 0.0
      epoch_psnr = 0.0

      with torch.set_grad_enabled(phase == 'train'):
        for i, data in enumerate(dataloaders[phase], 0):
          if phase == 'train':
            model.train()
            # try to set the gradient to None
            model.zero_grad(set_to_none = True)
            optimizer.zero_grad(set_to_none = True)
            params = training_params
          else:
            model.eval()
            params = val_params

          img, imgn, stdn_var, noise = create_input_variables(args, data)

          # Evaluate model
          try:
            with autocast(device_type='cuda'):
              pred = model(imgn, stdn_var)
              loss = compute_loss(criterion, pred, noise, imgn).cpu()
          except NameError:
            pred = model(imgn, stdn_var)
            loss = compute_loss(criterion, pred, noise, imgn).cpu()

          if phase == 'train':
            loss.backward()
            optimizer.step()
            model.eval()

            try:
              with autocast(device_type='cuda'):
                pred = model(imgn, stdn_var)
            except NameError:
              pred = model(imgn, stdn_var)

          out = torch.clamp(imgn - pred, 0., 1.)
          psnr = batch_psnr(out, img, 1.)

          if params['step'] % args.save_every == 0:
            if phase == 'train':
              # Apply regularization by orthogonalizing filters
              if not training_params['no_orthog']:
                model.apply(svd_orthogonalization)

            # Log the scalar values
            writer.add_scalar('loss on {} data'.format(phase), loss.item(), params['step'])
            writer.add_scalar('PSNR on {} data'.format(phase), psnr, params['step'])
            #print("[epoch %d][%d/%d] %s loss: %.4f PSNR: %.4f" %(epoch+1, i+1, total_batch_number, phase, loss.item(), psnr))
          params['step'] += 1
          running_loss += loss.item()
          running_psnr += psnr
          pbar.update(1)

      epoch_loss = running_loss / total_batch_number
      epoch_psnr = running_psnr / total_batch_number
      print("[epoch %d] %s loss: %.4f PSNR: %.4f" %(epoch+1, phase, epoch_loss, epoch_psnr))
      metrics['loss'][phase] = epoch_loss
      metrics['psnr'][phase] = epoch_psnr
      pbar.close()
    scheduler.step(metrics['loss']['val'])
    training_params['num_bad_epochs'] = scheduler.num_bad_epochs
    if get_lr(optimizer) != args.lr:
      args.lr = get_lr(optimizer)
      training_params['no_orthog'] = True
    writer.add_scalars('Loss', metrics['loss'], epoch)
    writer.add_scalars('PSNR', metrics['psnr'], epoch)
    writer.add_scalar('Learning rate', args.lr, epoch)

    # Log val images
    try:
      if epoch == 0:
        # Log graph of the model
        writer.add_graph(model, (imgn, stdn_var), )
        # Log validation images
        for idx in range(2):
          imclean = make_grid(img.data[idx].clamp(0., 1.), nrow=2, normalize=False, scale_each=False)
          imnsy = make_grid(imgn.data[idx].clamp(0., 1.), nrow=2, normalize=False, scale_each=False)
          writer.add_image('Clean validation image {}'.format(idx), imclean, epoch)
          writer.add_image('Noisy validation image {}'.format(idx), imnsy, epoch)
      for idx in range(2):
        imrecons = make_grid(out.data[idx].clamp(0., 1.), nrow=2, normalize=False, scale_each=False)
        writer.add_image('Reconstructed validation image {}'.format(idx), imrecons, epoch)
      # Log training images
      imclean = make_grid(img.data, nrow=8, normalize=True, scale_each=True)
      writer.add_image('Training patches', imclean, epoch)

    except Exception as e:
      logger.error("Couldn't log results: {}".format(e))

    # save model and checkpoint
    training_params['start_epoch'] = epoch + 1
    torch.save(model.state_dict(), os.path.join(args.experiment_name, 'net.pth'))
    if val_params['best_loss'] >  metrics['loss']['val']:
      val_params['best_loss'] = metrics['loss']['val']
      torch.save(model.state_dict(), os.path.join(args.experiment_name, 'best.pth'))
    save_dict = {
      'state_dict': model.state_dict(),
      'optimizer' : optimizer.state_dict(),
      'training_params': training_params,
      'val_params': val_params,
      'args': args
      }
    torch.save(save_dict, os.path.join(args.experiment_name, 'ckpt.pth'))
    if epoch % args.save_every_epochs == 0:
      torch.save(save_dict, os.path.join(args.experiment_name, 'ckpt_e{}.pth'.format(epoch+1)))
    del save_dict