from functions import*
import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch flower classification')

parser.add_argument('--epochs', default=3, type=int,help='number of total epochs to run')

parser.add_argument('--learning_rate', default=0.001, type=float, help='initial learning rate')

parser.add_argument('--hidden_units', type=int,default=1024, help='number of hidden units in the classifier')

parser.add_argument('--save-dir', dest='save_dir', help='The directory used to save the trained models',default='/', type=str)

parser.add_argument('--arch', default='resnet101',)

parser.add_argument('--data_dir', default='flowers')

parser.add_argument('--gpu',action='store_true')






def main():
    #building the model
    args = parser.parse_args()
    if(args.gpu):
        if(torch.cuda.is_available()):
            device = torch.device("cuda:0")
        else:
            print("GPU not available")
            return
    else:
        device = torch.device("cpu")
                  
    model,optimizer = building_model(args.hidden_units,args.arch,args.learning_rate)
    
    #Preparing data
    train_datasets,trainloader,validloader,testloader = data_preparation(args.data_dir)
    
    #training and testing the model
    
    train(model, trainloader,validloader,testloader,optimizer,device,args.epochs,args.hidden_units,/
          args.arch,args.learning_rate,args.save-dir)

if __name__ == "__main__":
    main()  
    
