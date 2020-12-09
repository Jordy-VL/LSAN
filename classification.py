from attention.model import StructuredSelfAttention
from attention.train import train
from losses import AsymmetricLossOptimized
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


def multilabel_classification(
    attention_model, train_loader, test_loader, epochs, GPU=True, criterion=torch.nn.BCELoss()
):
    opt = torch.optim.Adam(attention_model.parameters(), lr=0.001, betas=(0.9, 0.99))
    model = train(attention_model, train_loader, test_loader, criterion, opt, epochs, GPU)
    torch.save(model, "model.bin")
    return model


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


def get_losses(criterion):
    # start with simple CE, and make sure you reproduce your BCEloss results:
    loss_function = AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0)

    # (2) than try simple focal loss:
    loss_function = AsymmetricLoss(gamma_neg=2, gamma_pos=2, clip=0)

    # (3) try now ASL:
    loss_function = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0)
    loss_function = AsymmetricLoss(gamma_neg=4, gamma_pos=1, clip=0.05)

    # (4) also try the 'disable_torch_grad_focal_loss' mode, it can stabilize results:
    loss_function = AsymmetricLoss(
        gamma_neg=4, gamma_pos=1, clip=0.05, disable_torch_grad_focal_loss=True
    )
    return loss_function


def main(train, evaluate, criterion):

    if args.train:
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

        loss_fx = {
            "binary_crossentropy": torch.nn.BCELoss(),
            "assymetric_loss": AsymmetricLossOptimized(gamma_neg=4, gamma_pos=1, clip=0.05),
        }.get(criterion)

        (1)

        config["epochs"] = 4
        current_model = multilabel_classification(
            attention_model, train_loader, test_loader, epochs=config["epochs"], criterion=loss_fx
        )

    if args.evaluate:
        if args.train:
            model = current_model
        else:
            model = torch.load("model.bin")  # or it might not load all weights?
        predictions, golds = evaluate_after_loading(model, test_loader, loss_fx)
        evaluate_multilabel(predictions, golds, list(range(label_num)))

    # voc2idx = word_to_id
    # example = "Many methods for image captioning rely on pretrained object classifier CNN and Long Short Term Memory recurrent networks"
    # tokenized = example.lower().split()
    # encoded = np.vectorize(voc2idx.get)(tokenized)
    # torched = torch.from_numpy(encoded).type(torch.LongTensor).unsqueeze(0)

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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser("""Some customization""")
    parser.add_argument("-t", dest="train", action="store_true", default=True, help="train")
    parser.add_argument("-e", dest="evaluate", action="store_true", default=True, help="evaluate")
    parser.add_argument(
        "-l", dest="loss", default="binary_crossentropy", help="loss function from dict"
    )
    # parser.add_argument("-e", dest="ensembler", action="store_true", default=False, help="average over ensemble models")
    # parser.add_argument("-m", dest="ensemblesize", type=int, default=5, help="average over M ensemble models")

    args = parser.parse_args()

    main(args.train, args.evaluate, args.loss)
