import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import tensorflow as tf
import torch

from losses import *

# CONVERT TO TF
# CONVERT TO NER
'''
class AsymmetricLoss(nn.Module):
    def __init__(
        self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False
    ):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = x  # torch.sigmoid(x)
        xs_pos = x
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()
'''

def ASLSingleLabelTF_full(gamma_pos=0, gamma_neg=4, eps=0.1, reduction="mean"): #ASLSingleLabelTF
    #https://medium.com/@Bloomore/how-to-write-a-custom-loss-function-with-additional-arguments-in-keras-5f193929f7a0

    def internal_loss(y_true,y_pred):
        log_preds = tf.nn.log_softmax(y_pred, axis=-1) #like softmax

        """
        => Categorical encoding for labels; can assume this is already done
        """
        #scatter_(dim, index, src) 
        # tensor_scatter_add(tf.zeros(shape, values.dtype), indices, values)
        #tf.scatter_nd(indices, updates, shape)
        #ensor tells us that parameters include the dim, index tensor, and the source tensor.
        #Scatter updates into a new tensor according to indices.
        #tf.zeros_like(y_pred)
        # tf.expand_dims(tf.cast(y_true, tf.int64), 1)
        # targets_classes = tf.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
        # targets = targets_classes

        # ASL weights
        anti_targets = 1 - y_true
        xs_pos = tf.math.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * y_true
        xs_neg = xs_neg * anti_targets

        asymmetric_w = tf.math.pow(
            1 - xs_pos - xs_neg, gamma_pos * y_true + gamma_neg * anti_targets
        )
        log_preds = log_preds * asymmetric_w

        if eps > 0:  # label smoothing
            num_classes = y_true.shape[-1]
            y_true = tf.math.add(tf.math.multiply(y_true, 1-eps),(eps / num_classes))

        # loss calculation
        loss = tf.math.multiply(y_true, log_preds)
        loss = tf.math.reduce_sum(loss, axis=-1)

        if reduction == "mean":
            loss = tf.math.reduce_mean(loss)
        return loss    

def ASLSingleLabelTF(y_true,y_pred,gamma_pos=0, gamma_neg=4, eps=0.1, reduction="mean"):
    log_preds = tf.nn.log_softmax(y_pred, axis=-1) #like softmax

    """
    => Categorical encoding for labels; can assume this is already done
    """
    #scatter_(dim, index, src) 
    # tensor_scatter_add(tf.zeros(shape, values.dtype), indices, values)
    #tf.scatter_nd(indices, updates, shape)
    #ensor tells us that parameters include the dim, index tensor, and the source tensor.
    #Scatter updates into a new tensor according to indices.
    #tf.zeros_like(y_pred)
    # tf.expand_dims(tf.cast(y_true, tf.int64), 1)
    # targets_classes = tf.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)
    # targets = targets_classes

    # ASL weights
    anti_targets = 1 - y_true
    xs_pos = tf.math.exp(log_preds)
    xs_neg = 1 - xs_pos
    xs_pos = xs_pos * y_true
    xs_neg = xs_neg * anti_targets

    asymmetric_w = tf.math.pow(
        1 - xs_pos - xs_neg, gamma_pos * y_true + gamma_neg * anti_targets
    )
    log_preds = log_preds * asymmetric_w

    if eps > 0:  # label smoothing
        num_classes = y_true.shape[-1]
        y_true = tf.math.add(tf.math.multiply(y_true, 1-eps),(eps / num_classes))

    # loss calculation
    loss = tf.math.multiply(y_true, log_preds)
    loss = tf.math.reduce_sum(loss, axis=-1)

    if reduction == "mean":
        loss = tf.math.reduce_mean(loss)
    return loss    



def AsymmetricLossTF(y_true, y_pred, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8):
    """"
    Parameters
    ----------
    y_pred: input logits
    y: targets (multi-label binarized vector)
    """
    # Calculating Probabilities
    xs_pos = y_pred
    xs_neg = 1 - xs_pos

    # Asymmetric Clipping
    if clip is not None and clip > 0:
        xs_neg = tf.clip_by_value((xs_neg + clip),clip_value_min=eps,clip_value_max=1)

    # Basic CE calculation
    los_pos = y_true * tf.math.log(tf.clip_by_value(xs_pos,clip_value_min=eps,clip_value_max=float('inf')))
    los_neg = (1 - y_true) * tf.math.log(tf.clip_by_value(xs_neg,clip_value_min=eps,clip_value_max=float('inf')))
    loss = los_pos + los_neg

    # Asymmetric Focusing
    if gamma_neg > 0 or gamma_pos > 0:
        pt0 = xs_pos * y_true
        pt1 = xs_neg * (1 - y_true)  # pt = p if t > 0 else 1-p
        pt = pt0 + pt1
        one_sided_gamma = gamma_pos * y_true + gamma_neg * (1 - y_true)
        one_sided_w = tf.math.pow(1 - pt, one_sided_gamma)
        loss *= one_sided_w

    return tf.math.reduce_sum(loss)


def test_singlelabel_losses():
    # TENSORFLOW
    np.random.seed(42)
    torch.manual_seed(42)
    n_classes = 5
    sample_batch_x = np.random.random((32, n_classes))
    sample_batch_y = np.random.choice(
        list(range(n_classes)), size=(32,), p=[1 / n_classes for x in range(n_classes)]
    )
    tf_true = tf.keras.utils.to_categorical(sample_batch_y, num_classes=n_classes, dtype="float32")

    torch_pred = torch.from_numpy(sample_batch_x)
    torch_true = torch.from_numpy(sample_batch_y)
    pt_loss = ASLSingleLabel(gamma_pos=0, gamma_neg=4, eps=0.1, reduction="mean")
    torch_loss = pt_loss(torch_pred, torch_true)
    print("Torch: ", torch_loss)

    tf_pred = tf.convert_to_tensor(sample_batch_x, dtype=tf.float32)
    tf_loss = ASLSingleLabelTF(tf_true, tf_pred,gamma_pos=0, gamma_neg=4, eps=0.1, reduction="mean")
    print("TF: ", tf_loss)


def test_multilabel_losses():
    # TENSORFLOW
    np.random.seed(42)
    torch.manual_seed(42)
    n_classes = 5
    sample_batch_x = np.random.random((32, n_classes))
    sample_batch_y = np.random.choice(
        list(range(n_classes)), size=(32,), p=[1 / n_classes for x in range(n_classes)]
    )
    tf_true = tf.keras.utils.to_categorical(sample_batch_y, num_classes=n_classes, dtype="float32")

    torch_pred = torch.from_numpy(sample_batch_x)
    torch_true = torch.from_numpy(tf_true)
    pt_loss = AsymmetricLoss(gamma_neg=2, gamma_pos=2, clip=0)
    torch_loss = pt_loss(torch_pred, torch_true)
    print("Torch: ", torch_loss)

    tf_pred = tf.convert_to_tensor(sample_batch_x, dtype=tf.float32)
    tf_loss = AsymmetricLossTF(tf_true, tf_pred,gamma_neg=2, gamma_pos=2, clip=0)
    print("TF: ", tf_loss)



def generate_sequence_data_labels(
    samples=1028, vocab_size=74, max_len=80, unique=1, sequence_data=False
):
    # from pdb import set_trace
    # set_trace()
    sequences = np.eye(vocab_size)[np.random.randint(0, 30, (samples, max_len))]
    if sequence_data:
        labels = np.random.choice([0.0, 1.0], size=(samples, max_len, unique), p=[3 / 4, 1 / 4])
    else:
        labels = np.random.choice([0.0, 1.0], size=(samples, unique), p=[1 / 2, 1 / 2])
    return sequences, labels


def test_ner_losses():
    np.random.seed(42)
    torch.manual_seed(42)
    n_classes = 3
    sample_batch_x = tf.convert_to_tensor(np.random.random((1, 10, n_classes)), dtype=tf.float32)
    sample_batch_y = tf.convert_to_tensor(
        np.random.choice(
            list(range(n_classes)), size=(1, 10), p=[1 / n_classes for x in range(n_classes)]
        ),
        dtype=tf.int32,
    )


test_singlelabel_losses()
test_multilabel_losses()
