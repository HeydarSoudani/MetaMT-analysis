import os, argparse, torch, logging, warnings, time
import numpy as np
import cca_core

from torch.utils.data import DataLoader
from data import CorpusQA, CorpusSC, CorpusTC, CorpusPO, CorpusPA
from model import BertMetaLearning
from datapath import get_loc
from transformers import AutoModel


parser = argparse.ArgumentParser()
parser.add_argument("--first_model", type=str, default="", help="")
parser.add_argument("--second_model", type=str, default="", help="")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--seed", type=int, default=0, help="seed for numpy and pytorch")
parser.add_argument(
    "--model_name",
    type=str,
    default="xlm-roberta-base",
    help="name of the pretrained model",
)
parser.add_argument("--data_dir", type=str, default="data/", help="directory of data")
parser.add_argument("--task", type=str, default="sc_fa")
parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="")
parser.add_argument("--hidden_dims", type=int, default=768, help="")
parser.add_argument(
    "--local_model", action="store_true", help="use local pretrained model"
)
parser.add_argument("--sc_labels", type=int, default=3, help="")
parser.add_argument("--qa_labels", type=int, default=2, help="")
parser.add_argument("--tc_labels", type=int, default=10, help="")
parser.add_argument("--po_labels", type=int, default=18, help="")
parser.add_argument("--pa_labels", type=int, default=2, help="")
parser.add_argument("--qa_batch_size", type=int, default=8, help="batch size")
parser.add_argument("--sc_batch_size", type=int, default=32, help="batch size")
parser.add_argument("--tc_batch_size", type=int, default=32, help="batch size")
parser.add_argument("--po_batch_size", type=int, default=32, help="batch_size")
parser.add_argument("--pa_batch_size", type=int, default=8, help="batch size")


args = parser.parse_args()
print(args)

### == Device =====================
if torch.cuda.is_available():
  if not args.cuda:
    args.cuda = True
  torch.cuda.manual_seed_all(args.seed)
DEVICE = torch.device("cuda" if args.cuda else "cpu")

### == Randomness =================
torch.manual_seed(args.seed)
np.random.seed(args.seed)


### == Load Models ================
assert args.first_model != "", "Set first model"
assert args.second_model != "", "Set second model"
 
# first_model = BertMetaLearning(args).to(DEVICE)
# second_model = BertMetaLearning(args).to(DEVICE)
first_model = AutoModel.from_pretrained(
  args.model_name, local_files_only=args.local_model
).to(DEVICE)
second_model = AutoModel.from_pretrained(
  args.model_name, local_files_only=args.local_model
).to(DEVICE)

if args.first_model != "":
  first_model = torch.load(args.first_model)
if args.second_model != "":
  second_model = torch.load(args.second_model)

# print(first_model)
# for name, param in first_model.named_parameters():
#   print('name: {}, param: {}'.format(name, param.shape))
# print(second_model.clf_model.roberta.encoder.layer[0].attention.output.dense.weight.shape)


### == load Data ===================
def load_data(task_lang):
  [task, lang] = task_lang.split("_")
  if task == "qa":
      test_corpus = CorpusQA(
          *get_loc("test", task_lang, args.data_dir),
          model_name=args.model_name,
          local_files_only=args.local_model
      )
      batch_size = args.qa_batch_size
  elif task == "sc":
      test_corpus = CorpusSC(
          *get_loc("test", task_lang, args.data_dir),
          model_name=args.model_name,
          local_files_only=args.local_model
      )
      batch_size = args.sc_batch_size
  elif task == "tc":
      test_corpus = CorpusTC(
          get_loc("test", task_lang, args.data_dir)[0],
          model_name=args.model_name,
          local_files_only=args.local_model,
      )
      batch_size = args.tc_batch_size
  elif task == "po":
      test_corpus = CorpusPO(
          get_loc("test", task_lang, args.data_dir)[0],
          model_name=args.model_name,
          local_files_only=args.local_model,
      )
      batch_size = args.po_batch_size
  elif task == "pa":
      test_corpus = CorpusPA(
          get_loc("test", task_lang, args.data_dir)[0],
          model_name=args.model_name,
          local_files_only=args.local_model,
      )
      batch_size = args.pa_batch_size

  return test_corpus, batch_size

test_corpus, batch_size = load_data(args.task)
test_dataloader = DataLoader(
  test_corpus, batch_size=batch_size, pin_memory=True, drop_last=True
)
# batch = next(iter(test_dataloader))


cca_sim = []
first_model.eval()
second_model.eval()

all_f_acts1 = [[] for _ in range(13)]
all_f_acts2 = [[] for _ in range(13)]

with torch.no_grad():
  for batch in test_dataloader:
    batch["label"] = batch["label"].to(DEVICE)

    first_output = first_model.forward("sc", batch)
    second_output = second_model.forward("sc", batch)

    for i in range(len(first_output.hidden_states)):
      f_acts1 = torch.squeeze(first_output.hidden_states[i][:, 0, :], 1)  #[500, 768]
      f_acts2 = torch.squeeze(second_output.hidden_states[i][:, 0, :], 1) #[500, 768]

      all_f_acts1[i].append(f_acts1.cpu().detach().numpy())
      all_f_acts2[i].append(f_acts2.cpu().detach().numpy())

  all_f_acts1 = [np.concatenate(all_f_acts1[i]) for i in range(13)]
  all_f_acts2 = [np.concatenate(all_f_acts2[i]) for i in range(13)]
    

  results = []
  for i in range(13):
    f_results = cca_core.get_cca_similarity(all_f_acts1[i].T, all_f_acts2[i].T, epsilon=1e-10, verbose=False)
    results.append(f_results["cca_coef1"].mean())

print(results)





# # plot confusion matrix
#   df_cm = pd.DataFrame(cm, labels, labels)
#   sn.set(font_scale=1.4) # for label size
#   sn.heatmap(df_cm, annot=True, annot_kws={"size": 16}) # font size
#   plt.show()

