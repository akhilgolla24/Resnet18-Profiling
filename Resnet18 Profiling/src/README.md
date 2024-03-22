 Running this with the command:

 python3 lab2.py 

 with the followng options:

 optional arguments:
  -h, --help            show this help message and exit
  -c, --cuda            device, cuda or cpu
  -d DATAPATH, --datapath DATAPATH
                        path to dataset
  -w NUMWORKERS, --numworkers NUMWORKERS
                        num workers for dataloader
  -o OPTIMIZER, --optimizer OPTIMIZER
                        training loop optimizer, use sgd, adam, nest, adagrad, or adadelta
  -n, --nonorm            do not include batchnorm layer
  -s, --nodownload      do not download data

Explanation of options
 -c or --cuda, you can set the model to train on GPU (default is cpu)

 -d or --datapath allow the user to specify data path, default is the current working directory

 -w or --numworkers will allow the user to spcify the number of workers  for the dataloader, (default is 2)
  
 -o or --optimizer will allow the user to select sgd, adam, nest, adagrad, or adadelta as the optimiser (defautl is sgd)
    (nest is sgd with nesterov momentum)

-n or --nonorm, use this to build model without bacthnorm layers

-s or --nodownload, choose to download data if not already downloaded (will download by default)

Example calls:

python3 lab2.py -c -w 4 -o adagrad
trains model on gpu with 4 workers in dataloader using adagrad optimezer

python3 lab2.py -c -n -d ~/data
trains model on GPU without normalization using the dataset at ~/data

Using these options, you will be able to run any of the experiments defined in the homework.

OUTPUT:

The program will out put:

information about the options selected
ex. Optimizer= adagrad

Per epoch loss and accurracy:
ex.
Epoch 0: Loss= 2.168331 , Accuracy= 0.1714875000
Epoch 1: Loss= 1.875100 , Accuracy= 0.2873965000
Epoch 2: Loss= 1.664291 , Accuracy= 0.3792005000
Epoch 3: Loss= 1.460457 , Accuracy= 0.4697290000
Epoch 4: Loss= 1.310847 , Accuracy= 0.5333567500

data, train, and total times per epoch:
ex.
Total times per epoch: ['15.671945', '15.016071', '15.008815', '15.158497', '15.029000']
Data times per epoch: ['3.715291', '3.925823', '3.896245', '4.047680', '3.877618']
Train times per epoch: ['11.401689', '10.568981', '10.585046', '10.584259', '10.615076']

times of all epochs summed:
ex.
Total Time: 75.884329
Data Time: 19.462658
Train Time: 53.755051

final model training accurracy:
ex.
Model Train Accuracy: 0.659240

and the model parameters and gradients;
ex.
Model Parameters: 15648586
Model Gradients: 71


