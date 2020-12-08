from attention.model import StructuredSelfAttention
from attention.train import train
import torch
import numpy as np
import pandas as pd
import data.utils as utils
import data_got
from tqdm import tqdm

config = utils.read_config("config.yml")
if config.GPU:
    torch.cuda.set_device(0)
print("loading data...\n")
label_num = 54
train_loader, test_loader, label_embed, embed, X_tst, word_to_id, Y_tst, Y_trn = data_got.load_data(
    batch_size=config.batch_size
)
"""
Loading resources
"""
label_embed = torch.from_numpy(label_embed).float()  # [L*256]
embed = torch.from_numpy(embed).float()
print("load done")


def save_model(attention_model):
    torch.save(attention_model, "model.bin")


def predict():
    torch.load("model.bin")
    # already expects word embeddings most probably?


def multilabel_classification(attention_model, train_loader, test_loader, epochs, GPU=True):
    loss = torch.nn.BCELoss()
    opt = torch.optim.Adam(attention_model.parameters(), lr=0.001, betas=(0.9, 0.99))
    model = train(attention_model, train_loader, test_loader, loss, opt, epochs, GPU)
    torch.save(model, "model.bin")
    return model


attention_model = StructuredSelfAttention(
    batch_size=config.batch_size,
    lstm_hid_dim=config["lstm_hidden_dimension"],
    d_a=config["d_a"],
    n_classes=label_num,
    label_embed=label_embed,
    embeddings=embed,
)

if config.use_cuda:
    attention_model.cuda()
config["epochs"] = 4
current_model = multilabel_classification(
    attention_model, train_loader, test_loader, epochs=config["epochs"]
)


voc2idx = word_to_id
example = "Many methods for image captioning rely on pretrained object classifier CNN and Long Short Term Memory recurrent networks"
tokenized = example.lower().split()
encoded = np.vectorize(voc2idx.get)(tokenized)
torched = torch.from_numpy(encoded).type(torch.LongTensor).unsqueeze(0)


"""
#https://discuss.pytorch.org/t/loading-saved-models-gives-inconsistent-results-each-time/36312/24
current_model.eval()
current_model.batch_size = 1
prediction = current_model(torched.cuda())[0]
predicted_label = np.where(prediction.cpu()>0.5)[0]
print(predicted_label, prediction.cpu().detach().numpy()[predicted_label])
#array([ 2])
"""
"""
Load model, predict the same
"""
"""
model = torch.load("model.bin")  # or it might not load all weights?
model.eval()
model.batch_size = 1
prediction2 = model(torched.cuda())[0]
predicted_label2 = np.where(prediction2.cpu() > 0.5)[0]
print(predicted_label2, prediction2.cpu().detach().numpy()[predicted_label2])
"""
#            "cs.lg": 2,   -> machine learning
#            "cs.cl": 15,  -> Computation and Language


def evaluate_after_loading(model, test_loader, criterion):
    model.eval()
    model.batch_size = test_loader.batch_size

    test_loss = []
    preds, golds = [], []
    for batch_idx, test in enumerate(tqdm(test_loader)):
        x, y = test[0].cuda(), test[1].cuda()
        val_y = model(x)
        loss = criterion(val_y, y.float()) / test_loader.batch_size
        labels_cpu = y.data.cpu().float()
        pred_cpu = val_y.data.cpu()

        preds.append(pred_cpu.numpy())
        golds.append(labels_cpu.numpy())
        # prec = precision_k(labels_cpu.numpy(), pred_cpu.numpy(), 5)
        # test_acc_k.append(prec)
        test_loss.append(float(loss))

    avg_test_loss = np.mean(test_loss)
    print("epoch %2d test end : avg_loss = %.4f" % (0, avg_test_loss))

    preds = np.array(preds)
    golds = np.array(golds)
    preds = preds.reshape((-1, preds.shape[-1]))
    golds = golds.reshape((-1, golds.shape[-1]))

    return preds, golds


def evaluate_multilabel(confidences, onehot_gold, labels, threshold=0.5):
    def multilabel_encode(target_names, indices):
        empty = np.zeros((indices.shape[0], len(target_names)), dtype=int)
        for i, index in enumerate(indices):
            empty[i, np.array(index, dtype=int)] = 1
        return empty

    from sklearn.metrics import (
        confusion_matrix,
        accuracy_score,
        f1_score,
        log_loss,
        classification_report,
        brier_score_loss,
        precision_recall_curve,
        roc_curve,
        auc,
        roc_auc_score,
        matthews_corrcoef,
        mean_squared_error,
        hamming_loss,
    )

    predicted = np.array([np.where(x > threshold)[0] for x in confidences])
    onehot_predicted = multilabel_encode(labels, predicted)

    metric_names = ["Acc", "MSE(↓)", "F1(m)", "F1(M)", "NLL(↓)", "hamming_loss(↓)"]
    metrics = [
        round(x, 4)
        for x in [
            accuracy_score(onehot_gold, onehot_predicted),
            mean_squared_error(onehot_gold, onehot_predicted),
            f1_score(onehot_gold, onehot_predicted, average="weighted"),
            f1_score(onehot_gold, onehot_predicted, average="macro"),
            log_loss(onehot_gold, confidences),
            hamming_loss(onehot_gold, onehot_predicted)
            # calculate_ECE(
            #     onehot_gold, confidences
            # ),  # if isinstance(groundtruth[0], np.integer) else calculate_ECE(indices_groundtruth, confidences)  # brier?
            # brier_multi(onehot_gold, confidences, label2idx),
            # auc.result().numpy(),
            # average_confidence(confidences),
            # exp_entropy(confidences),
        ]
    ]
    df = pd.DataFrame([metrics])
    df.columns = metric_names
    print(df.head())
    print(df.to_markdown())


predictions, golds = evaluate_after_loading(current_model, test_loader, torch.nn.BCELoss())
evaluate_multilabel(predictions, golds, list(range(label_num)))


"""
StructuredSelfAttention(
  (embeddings): Embedding(69399, 300)
  (label_embed): Embedding(54, 300)
  (lstm): LSTM(300, 300, batch_first=True, bidirectional=True)
  (linear_first): Linear(in_features=600, out_features=200, bias=True)
  (linear_second): Linear(in_features=200, out_features=54, bias=True)
  (weight1): Linear(in_features=600, out_features=1, bias=True)
  (weight2): Linear(in_features=600, out_features=1, bias=True)
  (output_layer): Linear(in_features=600, out_features=54, bias=True)
  (embedding_dropout): Dropout(p=0.3, inplace=False)
)
"""
