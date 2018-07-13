from functions import*
import argparse
import torch
import json

parser = argparse.ArgumentParser(description='PyTorch flower classification')

parser.add_argument('image_path')
parser.add_argument('checkpoint')

parser.add_argument('--topk', type=int,default=1)

parser.add_argument('--gpu',action='store_true')
parser.add_argument('--category_names',default='cat_to_name.json')


def main():
   
  args = parser.parse_args()
    
  with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
        
  if(args.gpu):
        if(torch.cuda.is_available()):
            device = torch.device("cuda:0")
        else:
            print("GPU not available")
            return
  else:
           device = torch.device("cpu")
   
  model,class_to_idx = load_checkpoint(args.checkpoint)
  prob, classes = predict(args.image_path, model, device,class_to_idx,args.topk)
  max_index=np.argmax(prob)
  name=[cat_to_name[x] for x in classes]
  print(name)
  print([float(100*x) for x in prob])
if __name__ == "__main__":
    main()  
