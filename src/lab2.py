import torch
import torch.nn as nn
from torchvision.transforms import v2
from torchvision import datasets
# import numpy as np
# import sys
import argparse
import time

# import 

# write clases for building blocks w/ init and forward function
# create 8x building blocks, 2 of each sub group

class Resnet_Block(nn.Module):
    # 2 conv layers with relu
    def __init__(
            self,
            input_channels,
            output_channels,
            kernel,
            stride,
            padding,
            norm_layers = True
            ):
        super(Resnet_Block, self).__init__()
        self.norm_layers = norm_layers
        self.conv1 = nn.Conv2d(input_channels, output_channels, kernel, stride, padding, bias=False)
        self.norm1 = nn.BatchNorm2d(output_channels)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel, 1, padding, bias=False)
        self.norm2 = nn.BatchNorm2d(output_channels)

        self.identity = False
        self.identity_conv = None
        self.indentity_norm = None

        if stride != 1 or input_channels != output_channels:
            self.identity = True
            self.identity_conv = nn.Conv2d(input_channels, output_channels, kernel, stride, padding, bias=False)
            self.indentity_norm = nn.BatchNorm2d(output_channels)

        self.relu2 = nn.ReLU()

    def forward(self, x):
        if self.norm_layers == True:
            out = self.relu1(self.norm1(self.conv1(x)))
            out = self.norm2(self.conv2(out))
            # transform x?
            if self.identity:
                out += self.indentity_norm(self.identity_conv(x))
            else:
                out += x
            out = self.relu2(out)
            return out
        else:
            out = self.relu1(self.conv1(x))
            out = self.conv2(out)
            # transform x?
            if self.identity:
                out += self.identity_conv(x)
            else:
                out += x
            out = self.relu2(out)
            return out
        

class Resnet_18(nn.Module):
    def __init__(self, norm_layers=True):
        super(Resnet_18, self).__init__()
        # first layer
        self.norm_layers= norm_layers
        self.conv1 = nn.Conv2d(3, 64, (3,3), 1, 1, bias=False)
        self.norm1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        # 8 blocks
        self.block1 = Resnet_Block(64, 64, (3,3), 1, 1, norm_layers)
        self.block2 = Resnet_Block(64, 64, (3,3), 1, 1, norm_layers)

        self.block3 = Resnet_Block(64, 128, (3,3),2,1, norm_layers)
        self.block4 = Resnet_Block(128, 128, (3,3),2,1, norm_layers)
        
        self.block5 = Resnet_Block(128, 256, (3,3),2,1, norm_layers)
        self.block6 = Resnet_Block(256, 256, (3,3),2,1, norm_layers)

        self.block7 = Resnet_Block(256, 512, (3,3),2,1, norm_layers)
        self.block8 = Resnet_Block(512, 512, (3,3),2,1, norm_layers)

        self.blocks = nn.ModuleList()
        self.blocks.append(self.block1)
        self.blocks.append(self.block2)
        self.blocks.append(self.block3)
        self.blocks.append(self.block4)
        self.blocks.append(self.block5)
        self.blocks.append(self.block6)
        self.blocks.append(self.block7)
        self.blocks.append(self.block8)


        # final classification layer
        self.flat1 = nn.Flatten()
        self.lin1 = nn.Linear(512, 10)

    def forward(self, x):
        if self.norm_layers == True:
            x = self.relu1(self.norm1(self.conv1(x)))
        else:
            x = self.relu1(self.conv1(x))
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.lin1(self.flat1(x))
        return x

def cifar_loader(data_path = './data', num_workers=2, download_data = True):
    transform = v2.Compose([
                v2.ToImage(), 
                v2.ToDtype(torch.float32, scale=True),
                v2.RandomCrop(size= (32,32), padding=4),
                v2.RandomHorizontalFlip(p=0.5),
                v2.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
                ])


    trainset = datasets.CIFAR10(root=data_path, train=True, download=download_data, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=num_workers)

    return trainloader
    

def train(model, dataloader, epochs=5, optmizer_type = 'sgd', device='cpu'):
    model.train()

    optimizer = None

    if optmizer_type == 'nest':
        print("Optimizer= "+ optmizer_type)
        optimizer = torch.optim.SGD(list(model.parameters()), lr=0.1, momentum=0.9, weight_decay= .0005, nesterov=True)

    elif optmizer_type == 'adagrad':
        print("Optimizer= "+ optmizer_type)
        optimizer = torch.optim.Adagrad(list(model.parameters()), lr=0.1, weight_decay=0.0005)

    elif optmizer_type == 'adadelta':
        print("Optimizer= "+ optmizer_type)
        optimizer = torch.optim.Adadelta(list(model.parameters()), lr=0.1, weight_decay=0.0005)
        
    elif optmizer_type == 'adam':
        print("Optimizer= "+ optmizer_type)
        optimizer = torch.optim.Adam(list(model.parameters()), lr=0.1, weight_decay=0.0005)

    else:
        print("Optimizer= "+ optmizer_type)
        optimizer = torch.optim.SGD(list(model.parameters()), lr=0.1, momentum=0.9, weight_decay= .0005)

    criterion = nn.CrossEntropyLoss()
    
    epoch_total_times = [0] * 5
    epoch_dataloading_times = [0] * 5
    epoch_training_times = [0] * 5
    epoch_loss = [0] * 5
    epoch_acc = [0] * 5


    if device == "cuda":
        torch.cuda.synchronize()
    epoch_start = time.monotonic_ns()

    for epoch in range(epochs):
        model.train()
        batch = 0 
        if device == "cuda":
            torch.cuda.synchronize()
        data_start = time.monotonic_ns()
        for x, y in dataloader:
            
            # dataloading time
            if device == "cuda":
                torch.cuda.synchronize()
            data_end = time.monotonic_ns()
            epoch_dataloading_times[epoch] += data_end - data_start

            batch+=1
            
            # move data to gpu
            if device == 'cuda':
                x = x.to(device=device)
                y = y.to(device=device)
            
            
           # start training 
            if device == "cuda":
                torch.cuda.synchronize()
            train_start =  time.monotonic_ns()
            optimizer.zero_grad()
            y_pred = model(x)
            loss = criterion(y_pred, y)

            
            loss.backward()
            optimizer.step()
            # end training
            if device == "cuda":
                torch.cuda.synchronize()
            train_end = time.monotonic_ns()
            epoch_training_times[epoch] += train_end - train_start
            
            # calculate acc and loss 

            _, y_pred_label = torch.max(y_pred, 1)

            correct_count = (y_pred_label == y).sum().item()
            samples_count = y.size(0)
            acc = correct_count/samples_count

            print("Epoch " + str(epoch) + ", Batch "+ str(batch) + ", Loss: " + str('%.6f' % loss.item()) + " Acc: " + str('%.6f' % acc), end="\r")

            epoch_loss[epoch] += loss.item()
            epoch_acc[epoch] += acc

            # start data time
            if device == "cuda":
                torch.cuda.synchronize()
            data_start = time.monotonic_ns()

        # start epoch time
        if device == "cuda":
            torch.cuda.synchronize()
        end_time = time.monotonic_ns()
        epoch_total_times[epoch] = end_time-epoch_start

        # print epoch loss and acc
        epoch_loss[epoch] = epoch_loss[epoch] / batch
        epoch_acc[epoch] = epoch_acc[epoch] / batch
        print("Epoch " + str(epoch) + ": Loss= " + str('%.6f' % epoch_loss[epoch]) + " , Accuracy= " + str('%.6f' % epoch_acc[epoch]))

        # start next epoch
        epoch_start = time.monotonic_ns()


        # print(epoch)

    model.eval()
    return epoch_total_times, epoch_dataloading_times, epoch_training_times

def accuracy(model, dataloader, device):
    model.eval()

    total_correct = 0
    total_samples = 0
    for x, y in dataloader:
        if device == 'cuda':
            x = x.to(device=device)
            y = y.to(device=device)

        y_pred = model(x)
        _, y_pred_label = torch.max(y_pred, 1)
        total_correct += (y_pred_label == y).sum().item()
        total_samples += y.size(0)
    
    return total_correct/total_samples



if __name__ == "__main__":
    #params:
    # cuda: default = ??
    # datapath default = './data'
    # optimizer - default = sgd
    # num_workers - default = 2
    # measure for all experiments: measure time - training loop (dataloader, batch, epochpt)
    # measure for all experiments: measure time - total time (always?)
    # 
    parser = argparse.ArgumentParser()

    # default values
    data_path = './data'
    device = 'cpu'
    optimizer = 'sgd'
    num_workers = 2
    norm_layers = True
    download_data = True

    parser.add_argument('-c','--cuda', help='device, cuda or cpu', required=False, action='store_true')
    parser.add_argument('-d','--datapath', help='path to dataset', required=False)
    parser.add_argument('-w','--numworkers', help='num workers for dataloader', required=False)
    parser.add_argument('-o','--optimizer', help='training loop optimizer, use sgd, adam, nest, adagrad, or adadelta', required=False)
    parser.add_argument('-n','--nonorm', help='do not include batchnorm layer', required=False, action='store_true')
    parser.add_argument('-s','--nodownload', help='do not download data', required=False, action='store_true')

    args = vars(parser.parse_args())

    # use cuda
    if args['cuda'] is True and torch.cuda.is_available():
        device = 'cuda'
        print("Using GPU")
    elif args['cuda'] is True and torch.cuda.is_available() == False:
        print("cuda not available")

    # custom datapath
    if args['datapath'] is not None:
        data_path = args['datapath']

    # optimizer
    if args['optimizer'] is not None and args['optimizer'] in ['sgd', 'adam', 'nest', 'adagrad', 'adadelta']: # add other optimzers
        optimizer = args['optimizer']
        print("Optimizers = " + str(args['optimizer']))

    # numworkers
    if args['numworkers'] is not None:
        num_workers = int(args['numworkers'])
        print("Num workers = " + str(args['numworkers']))

    # include bath norm layers
    if args['nonorm'] is True:
        norm_layers = False
        print("normalize layers = " + str(norm_layers))

    if args['nodownload'] is True:
        download_data = False


    dataloader = cifar_loader(data_path, num_workers, download_data)
    model = Resnet_18(norm_layers)
    if device == 'cuda':
        model.to(device=device)

    print("Model Training started:")
    total_time, data_time, train_time = train(model, dataloader, 5, optimizer, device=device)

    print("")
    # format times times
    total_time_epoch = ['%.6f' % (t/1000000000) for t in total_time]
    data_time_epoch = ['%.6f' % (t/1000000000) for t in data_time]
    train_time_epoch = ['%.6f' % (t/1000000000) for t in train_time]

    print("Total times per epoch: " + str(total_time_epoch))
    print("Data times per epoch: " + str(data_time_epoch))
    print("Train times per epoch: " + str(train_time_epoch))
    print("")

    total_time_s = '%.6f' % (sum(total_time)/1000000000)
    data_time_s = '%.6f' % (sum(data_time)/1000000000)
    train_time_s = '%.6f' % (sum(train_time)/1000000000)

    print("Total Time: " + str(total_time_s))
    print("Data Time: " + str(data_time_s))
    print("Train Time: " + str(train_time_s))
    
    # eval model:
    model_acc = accuracy(model, dataloader, device)
    print("Model Train Accuracy: " + str('%.6f' % model_acc))

    # Q3/Q4 get model Parameters: 
    total_params = 0
    total_grad = 0
    for ele in model.parameters():
        total_params += ele.numel()
        if ele.requires_grad:
            total_grad += 1
    print()
    print("Model Parameters: " + str(total_params))
    print("Model Gradients: " + str(total_grad))
