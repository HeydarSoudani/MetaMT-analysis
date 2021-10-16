import os, argparse, torch, logging, warnings, time
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from transformers import AutoModel, AutoModelForSequenceClassification
from data import CorpusQA, CorpusSC, CorpusTC, CorpusPO, CorpusPA
from utils import evaluateQA, evaluateNLI, evaluateNER, evaluatePOS, evaluatePA
from sklearn.metrics import confusion_matrix, accuracy_score
from torch.utils.data import DataLoader
import torch.nn as nn

from datapath import get_loc
from model import BertMetaLearning
from data import CorpusSC
from utils import evaluateNLI

logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()

parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
parser.add_argument("--dropout", type=float, default=0.1, help="")
parser.add_argument("--hidden_dims", type=int, default=768, help="")

parser.add_argument(
    "--model_name",
    type=str,
    default="xlm-roberta-base",
    help="name of the pretrained model",
)
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

parser.add_argument("--seed", type=int, default=0, help="seed for numpy and pytorch")
parser.add_argument(
    "--log_interval",
    type=int,
    default=100,
    help="Print after every log_interval batches",
)
parser.add_argument("--data_dir", type=str, default="data/", help="directory of data")
parser.add_argument("--cuda", action="store_true", help="use CUDA")
parser.add_argument("--save", type=str, default="saved/", help="")
parser.add_argument("--load", type=str, default="", help="")
parser.add_argument("--grad_clip", type=float, default=1.0)

parser.add_argument("--task", type=str, default="qa_hi")

parser.add_argument("--n_best_size", default=20, type=int)
parser.add_argument("--max_answer_length", default=30, type=int)
parser.add_argument(
    "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
)
parser.add_argument("--warmup", default=0, type=int)
parser.add_argument(
    "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
)

args = parser.parse_args()
print(args)

logger = {"args": vars(args)}
logger["train_loss"] = []
logger["val_loss"] = []
logger["val_metric"] = []
logger["train_metric"] = []

torch.manual_seed(args.seed)
np.random.seed(args.seed)

if not os.path.exists(args.save):
    os.makedirs(args.save)

if torch.cuda.is_available():
    if not args.cuda:
        # print("WARNING: You have a CUDA device, so you should probably run with --cuda")
        args.cuda = True

    torch.cuda.manual_seed_all(args.seed)

DEVICE = torch.device("cuda" if args.cuda else "cpu")

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

class BertCCA(nn.Module):
  def __init__(self, args):
    super(BertCCA, self).__init__()
    self.args = args
    self.device = None

    self.clf_model = AutoModelForSequenceClassification.from_pretrained(
      args.model_name,
      num_labels=args.sc_labels,
      local_files_only=args.local_model,
    )

    # Sequence Classification
    self.sc_dropout = nn.Dropout(args.dropout)
    self.sc_classifier = nn.Linear(args.hidden_dims, args.sc_labels)

  def forward(self, task, data):

    if "sc" in task:
      data["input_ids"] = data["input_ids"].to(self.device)
      data["attention_mask"] = data["attention_mask"].to(self.device)
      data["token_type_ids"] = data["token_type_ids"].to(self.device)
      data["label"] = data["label"].to(self.device)

      outputs = self.clf_model(
        data["input_ids"],
        token_type_ids=data["token_type_ids"],
        attention_mask=data["attention_mask"],
        labels=data["label"],
        output_hidden_states=True
      )
      return outputs

  def to(self, *args, **kwargs):
    self = super().to(*args, **kwargs)
    self.device = args[0]  # store device
    self.clf_model = self.clf_model.to(*args, **kwargs)
    self.sc_dropout = self.sc_dropout.to(*args, **kwargs)
    self.sc_classifier = self.sc_classifier.to(*args, **kwargs)
    return self



# plot CCA similarity
def plot_cca_similarity():

  model = BertCCA(args).to(DEVICE)
  if args.load != "":
    model = torch.load(args.load)
  
  print(model)
  time.sleep(5)

  model.eval()
  with torch.no_grad():
    total_loss = 0.0
    correct = 0.0
    total = 0.0
    matrix = [[0 for _ in range(3)] for _ in range(3)]
    for batch in test_dataloader:
      batch["label"] = batch["label"].to(DEVICE)
      output = model.forward("sc", batch)
      print(output)


if __name__ == "__main__":
  plot_cca_similarity()




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

