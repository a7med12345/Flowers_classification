from functions import*
import argparse
import torch

parser = argparse.ArgumentParser(description='PyTorch flower classification')

parser.add_argument('image_path')
parser.add_argument('checkpoint')

parser.add_argument('--topk', type=int)

parser.add_argument('--gpu',action='store_true')
parser.add_argument('--category_names')


def main():
  args = parser.parse_args()
  if(args.gpu):
        if(torch.cuda.is_available()):
            device = torch.device("cuda:0")
        else:
            print("GPU not available")
            return
  else:
           device = torch.device("cpu")
   
  model,class_to_idx = load_checkpoint(args.checkpoint)
  print(predict(args.image_path, model, device,class_to_idx))
    
if __name__ == "__main__":
    main()  
