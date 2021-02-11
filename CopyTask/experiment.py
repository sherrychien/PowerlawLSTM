import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import pickle



class FunctionExperiment():
    """Experiment on Learning a function from input parameters"""

    def __init__(self, dataset, model, model_name, optimizer, metric=None, train_data_parameters=None, test_data_parameters=None,
                 batch_size=32, lr=1e-3, lr_schedule=None, max_epochs=20, patience=0, clip_gradients=None,
                 device='cpu', print_steps=500, save_path='.', name='CopyTaskExperiment'):
        super(FunctionExperiment, self).__init__()
        self.model = model
        self.model_name = model_name
        self.optimizer_type = optimizer
        self.dataset = dataset
        self.train_data_parameters = train_data_parameters
        self.test_data_parameters = test_data_parameters
        self.lr = lr
        self.lr_schedule = lr_schedule
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.clip_gradients = clip_gradients
        self.device = device
        self.patience = patience
        if metric is None:
            self.metric = nn.NLLLoss().to(device)
        else:
            self.metric = metric
        
        self.seq_length = self.train_data_parameters['seq_length']

        if self.optimizer_type is None:
            raise ValueError('Optimizer needs to be defined')
        if self.model is None:
            raise ValueError('Model needs to be defined')
        if self.dataset is None:
            raise ValueError('Dataset needs to be defined')

        self.print_steps = print_steps
        self.save_path = save_path
        self.loss_function = None
        self.optimizer = None
        self.epoch = 0
        self.total_params = 0

    def load_data(self, subset):
        if subset == 'train':
            data_instance = self.dataset(subset, device=self.device, **self.train_data_parameters)
        elif subset == 'valid' or subset == 'test':
            data_instance = self.dataset(subset, device=self.device, **self.test_data_parameters)
        return data_instance

    def model_setup(self):
        self.model.to(self.device)
        self.loss_function = nn.NLLLoss().to(self.device)
        self.optimizer = self.optimizer_type(self.model.parameters(), lr=self.lr, alpha = 0.9)

    def eval_model(self, dataloader):
        total_metric1 = 0.0
        total_metric2 = 0.0
        total_items = 0.0
        total_loss = 0.0
        total_batches = 0
        with torch.no_grad():
            for i_batch, sample_batched in enumerate(dataloader):
                source, targets = sample_batched['input'], sample_batched['output']
                batch_size = source.shape[0]
                # Predict for this batch
                scores, _ = self.model(source)
                # NOTE: Softmax missing at the output of the network
                targets = torch.squeeze(targets).long()
                accu1, accu2 = self.metric(scores, targets)
                val_loss = self.loss_function(scores.transpose(1, 2), targets)
                total_loss += val_loss.item()
                total_metric1 += accu1
                total_metric2 += accu2
                total_batches += 1
        return total_metric1 / total_batches, total_metric2 / total_batches, total_loss / total_batches

    def train_model(self):
        dataset_train = self.load_data("train")
        self.data_train = dataset_train
        dataloader_train = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=False, num_workers=0)

        dataset_val = self.load_data("valid")
        self.data_val = dataset_val
        dataloader_val = DataLoader(dataset_val, batch_size=self.batch_size, shuffle=False, num_workers=0)

        self.model_setup()
        self.total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        results = {
            'val_metric1': [],
            'val_metric2':[],
            'val_loss': [],
            'train_loss': [],
        }
        start_epoch = self.epoch
        batch_num = 100000//self.batch_size
        torch.autograd.set_detect_anomaly(True)

        for epoch in range(start_epoch, self.max_epochs):
            # Training
            print('Starting Epoch', epoch)
            train_loss = []
            for i_batch, sample_batched in enumerate(dataloader_train):
                self.model.train()
                source, targets = sample_batched['input'], sample_batched['output']
                # NOTE: We don't need to reset the initial hidden state because the default is to use zero for c0 and h0
                self.model.zero_grad()

                scores, _ = self.model(source)
                targets = torch.squeeze(targets).long()
                total_loss = self.loss_function(scores.transpose(1, 2), targets)
                total_loss.backward()

                if self.clip_gradients is not None:
                    _ = nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_gradients)
                self.optimizer.step()
                train_loss.append(total_loss.item())
                
                if i_batch % self.print_steps == 0:
                    print('Batch', i_batch, 'Loss:', total_loss.item(), 'mean loss', sum(train_loss) / (i_batch + 1))


            self.model.eval()
            accu1, accu2, total_val_loss = self.eval_model(dataloader_val)
            results['val_metric1'].append(accu1)
            results['val_metric2'].append(accu2)
            results['val_loss'].append(total_val_loss)
            results['train_loss'].append(train_loss)
            train_loss = []
            print('Accuracy1: ', accu1, 'Accuracy2:', accu2)
            print('Validation loss: ', total_val_loss)
            
            with open(self.save_path+'/CopyTask_%s.pickle'%(self.model_name), 'wb') as handle:
                pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)

            torch.save(self.model.state_dict(),self.save_path+'/CopyTask_%s.pt'%(self.model_name))


            if  accu1 >= (1.0-1e-3) and accu2 >= (1.0-1e-3):
                print('Convergence achieved at iteration ', epoch*len(dataloader_train)+i_batch)
                break
            elif epoch>200 and accu1<0.2:
                print('model fail to converge')
                break
          
            self.epoch = epoch

        return results


    def test_model(self):
        # Test the model
        results = {
        'test_metric1': 0.0,
        'test_metric2': 0.0,
        'total_loss': 0.0

        }
        self.model.eval()
        dataset_test = self.load_data("test")
        dataloader_test = DataLoader(dataset_test, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.model_setup()
        total_metric1, total_metric2, total_loss = self.eval_model(dataloader_test)
        print('Test metric1: ', total_metric1, 'Test metric2: ', total_metric2, 'total loss:', total_loss)
        results['test_metric1'] = total_metric1
        results['test_metric2'] = total_metric2
        results['total_loss'] = total_loss
        return results
