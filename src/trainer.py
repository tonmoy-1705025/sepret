import os
import time
import torch
from src.pit_criterion import cal_loss_pit, cal_loss_no, MixerMSE
from torch.utils.tensorboard import SummaryWriter
import gc


class Trainer(object):

    def __init__(self, data, model, optimizer, config):

        self.tr_loader = data["tr_loader"]
        self.cv_loader = data["cv_loader"]
        self.model = model
        self.optimizer = optimizer

        # Training config
        self.use_cuda = config["train"]["use_cuda"]  # Whether to use GPU
        self.epochs = config["train"]["epochs"]      # training batch
        self.half_lr = config["train"]["half_lr"]    # Whether to adjust the learning rate
        self.early_stop = config["train"]["early_stop"]  # Whether to stop early
        self.max_norm = config["train"]["max_norm"]  # L2 norm

        # save and load model
        self.save_folder = config["save_load"]["save_folder"]  # Model save path
        self.checkpoint = config["save_load"]["checkpoint"]  # Whether to save each trained model
        self.continue_from = config["save_load"]["continue_from"]  # Whether to continue the original training progress?
        self.model_path = config["save_load"]["model_path"]  # Model saving format

        # logging
        self.print_freq = config["logging"]["print_freq"]

        # loss
        self.tr_loss = torch.Tensor(self.epochs)
        self.cv_loss = torch.Tensor(self.epochs)

        # Generate a folder to save the model
        os.makedirs(self.save_folder, exist_ok=True)
        self.prev_val_loss = float("inf")
        self.best_val_loss = float("inf")
        self.halving = False
        self.val_no_improve = 0

        # Visualization
        self.write = SummaryWriter("./logs")

        self._reset()

    def _reset(self):
        if self.continue_from:
            # Then the original progress training
            print('Loading checkpoint model %s' % self.continue_from)
            package = torch.load(self.save_folder + self.continue_from)

            if isinstance(self.model, torch.nn.DataParallel):
                self.model = self.model.module

            self.model.load_state_dict(package['state_dict'])
            self.optimizer.load_state_dict(package['optim_dict'])

            self.start_epoch = int(package.get('epoch', 1))

            self.tr_loss[:self.start_epoch] = package['tr_loss'][:self.start_epoch]
            self.cv_loss[:self.start_epoch] = package['cv_loss'][:self.start_epoch]
        else:
            # retrain
            self.start_epoch = 0


    """
    Training
    """
    def train(self):

        print("##### @@ Train Start  @@ #####")
        print(self.start_epoch)

        for epoch in range(self.start_epoch, self.epochs):
            
            self.model.train()                                      # Set the model to training mode


            start_time = time.time()                                # Training start time
            print("------- Epoch Information( Training ) --------")
            tr_loss = self._run_one_epoch(epoch)                    # Training model
            self.write.add_scalar("train loss", tr_loss, epoch+1)
            print("----------- Training Loss ----------")
            print('Time : {0:.2f}s'.format(time.time() - start_time))
            print('Train Loss : {0:.3f}\n'.format(tr_loss))


            """
            Description : Save the model to checkpoint folder
            """
            # Save each trained model
            file_path = os.path.join(self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))

            if self.continue_from == "":
                if isinstance(self.model, torch.nn.DataParallel):
                    self.model = self.model.module

            torch.save(self.model.serialize(self.model,
                                            self.optimizer,
                                            epoch + 1,
                                            tr_loss=self.tr_loss,
                                            cv_loss=self.cv_loss), file_path)

            print('Saving checkpoint model to %s ..... \n' % file_path)

            
            """
            Description: Start Cross Validation
            """
            self.model.eval()                                           # Set the model to validation mode

            start_time = time.time()                                    # Verification start time

            print("------- Epoch Information( Cross Validation ) --------")
            val_loss = self._run_one_epoch(epoch, cross_valid=True)     # Verify model
            self.write.add_scalar("validation loss", val_loss, epoch+1)
            print("----------- Cross Validation Result ----------")
            print('Time : {0:.2f}s'.format(time.time()-start_time))
            print('Train Loss : {0:.3f}\n'.format(tr_loss))


            """
            Description: Whether to adjust the learning rate
            """
            if self.half_lr:
                # Verify whether the loss has increased
                if val_loss >= self.prev_val_loss:
                    self.val_no_improve += 1  # Count the number of times there has been no improvement

                    # If there is no improvement after training for 3 epochs, the learning rate is halved.
                    if self.val_no_improve >= 3:
                        self.halving = True

                    # If there is no improvement after training for 10 epochs, end training
                    if self.val_no_improve >= 10 and self.early_stop:
                        print("No improvement for 10 epochs, early stopping.")
                        break
                else:
                    self.val_no_improve = 0

            if self.halving:
                optime_state = self.optimizer.state_dict()
                optime_state['param_groups'][0]['lr'] = optime_state['param_groups'][0]['lr']/2.0
                self.optimizer.load_state_dict(optime_state)
                print('Learning rate adjusted to: {lr:.6f}'.format(lr=optime_state['param_groups'][0]['lr']))
                self.halving = False

            self.prev_val_loss = val_loss  # current loss

            self.tr_loss[epoch] = tr_loss
            self.cv_loss[epoch] = val_loss


            """
            Description: Save the best model
            """
            if val_loss < self.best_val_loss:

                self.best_val_loss = val_loss  # Minimum verification loss value

                file_path = os.path.join(self.save_folder, self.model_path)

                torch.save(self.model.serialize(self.model,
                                                self.optimizer,
                                                epoch + 1,
                                                tr_loss=self.tr_loss,
                                                cv_loss=self.cv_loss), file_path)

                print("Find better validated model, saving to %s" % file_path)

    """
    Run a single epoch
    """
    def _run_one_epoch(self, epoch, cross_valid=False):

        start_time = time.time()

        total_loss = 0
        data_loader = self.tr_loader if not cross_valid else self.cv_loader  # Dataset switching

        print(f"Epoch No.: {epoch+1}")
        print(f"Total Iteration : {len(data_loader)}")
        print('-' * 85)
        print("Epoch | Itration | Average-Loss | Current-Loss | Time (s/batch)")
        print('-' * 85)

        for i, (data) in enumerate(data_loader):

            padded_mixture, mixture_lengths, padded_source = data

            # Whether to use GPU training
            if torch.cuda.is_available():
                padded_mixture = padded_mixture.cuda()
                mixture_lengths = mixture_lengths.cuda()
                padded_source = padded_source.cuda()

            estimate_source = self.model(padded_mixture)  # Put data into model

            # Calculate loss
            loss, _, estimate_source, _ = cal_loss_pit(padded_source, estimate_source, mixture_lengths)

            # Check if it running Cross Validation
            if not cross_valid:
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_norm)
                self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                
                self.print_loss( "Epoch" + str(epoch+1), 
                                 i+1, 
                                '{0:.4f}'.format(total_loss/(i+1)), 
                                '{0:.4f}'.format(loss.item()),
                                '{0:.4f}'.format((time.time()-start_time)/(i+1)) )

                # print('Epoch {0} \t  {1} \t {2:.3f} \t {3:.3f} \t {4:.1f} '.format(
                #     epoch+1,
                #     i+1,
                #     total_loss/(i+1),
                #     loss.item(),
                #     (time.time()-start_time)/(i+1)),
                #     flush=True)

            # Clean Memory
            gc.collect()
            torch.cuda.empty_cache()

        return total_loss/(i+1)


    def print_loss(self,  epoch_no , iter_no , total_loss, curr_loss, time ):
        c1 = 9 - len(epoch_no)
        c2 = 9 - len(str(iter_no))
        c3 = 13 - len(str(total_loss))
        c4 = 15 - len(str(curr_loss))

        print(epoch_no , ' '*c1, iter_no ,' '*c2, total_loss , ' '*c3, curr_loss , ' '*c4, time)

