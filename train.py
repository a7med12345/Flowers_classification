from fuunction import*
import argparse

parser = argparse.ArgumentParser(description='PyTorch flower classification')

parser.add_argument('--epoch', default=3, type=int, metavar='N',
                    help='number of total epochs to run')

parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate')


parser.add_argument('--hidden_units', type=int,default=1024
                    help='number of hidden units in the classifier')

parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='/', type=str)

parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet101',
                    choices=model_names,help='model architecture: ' + ' | '.join(model_names) +' (default: vgg13)')


parser.add_argument('--gpu')






def main():
    #building the model
    args = parser.parse_args()
    if(args.gpu):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            print("GPU not available)
            return
     else:
        device = torch.device("cpu")
                  
    model = building_model(args.hidden_units,args.arch)
    
    #Preparing data
    train_datasets,trainloader,validloader,testloader = data_preparation(args.data_dir)
    
    #training and testing the model
    
    train(model, trainloader,validloader,testloader,arg.learning-rate,device,args.epochs)
    
    
