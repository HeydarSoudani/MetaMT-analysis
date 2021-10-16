import os, argparse, torch, logging, warnings, time
import numpy as np
import cca_core
from model import BertMetaLearning

parser = argparse.ArgumentParser()
parser.add_argument("--first_model", type=str, default="", help="")
parser.add_argument("--second_model", type=str, default="", help="")
args = parser.parse_args()
print(args)

### == Device =====================
if torch.cuda.is_available():
  if not args.cuda:
    args.cuda = True
  torch.cuda.manual_seed_all(args.seed)
DEVICE = torch.device("cuda" if args.cuda else "cpu")


### == Load Models ================
assert args.first_model != "", "Set first model"
assert args.second_model != "", "Set second model"
 
first_model = BertMetaLearning(args).to(DEVICE)
second_model = BertMetaLearning(args).to(DEVICE)

if args.first_model != "":
  first_model = torch.load(args.first_model)
if args.second_model != "":
  second_model = torch.load(args.second_model)


print(first_model.roberta.state_dict())
time.sleep(5)

cca_sim = []
for layer in range(12):
  f_acts1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8 ,9], [0, 0, 0]])
  f_acts2 = np.array([[1, 1, 1], [4, 5, 6], [7, 8 ,9], [0, 0, 0]])

  f_results = cca_core.get_cca_similarity(f_acts1.T, f_acts2.T, epsilon=1e-10, verbose=False)
  print(f_results["cca_coef1"].mean())
  cca_sim.append(f_results["cca_coef1"].mean())





if __name__ == "__main__":
  # plot_cca_similarity()
  pass




# # plot confusion matrix

# model = BertMetaLearning(args).to(DEVICE)
# if args.load != "":
#   model = torch.load(args.load)

# def plot_confusion_matrix():
#   model.eval()

#   if "qa" in args.task:
#     pass
#   elif "sc" in args.task:
#     _, _, cm = evaluateNLI(model, test_dataloader, DEVICE)
#     labels = ['contradiction', 'entailment', 'neutral']
#   elif "tc" in args.task:
#     pass
#   elif "po" in args.task:
#     pass
#   elif "pa" in args.task:
#     pass

#   df_cm = pd.DataFrame(cm, labels, labels)
#   sn.set(font_scale=1.4) # for label size
#   sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
#   plt.show()

