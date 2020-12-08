This is the code for our paper "Label-Specific Document Representation for Multi-Label Text Classification"   
If you make use of this code or the LSAN algorithm in your work, please cite the following paper:

@inproceedings{xiao2019label,  
  title={Label-Specific Document Representation for Multi-Label Text Classification},  
  author={Xiao, Lin and Huang, Xin and Chen, Boli and Jing, Liping},  
  booktitle={Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)},  
  pages={466--475},  
  year={2019}  
}  
 
 
Requirements: Ubuntu 16.0.4;  
  Python 3.6.7;  
  Pytorch 1.0.0;  
  mxnet 1.3.1;  
  
Reproducibility: We provide the processed dataset AAPD, put them in the folder./data/

Train: python classification.py

Processed data download: https://drive.google.com/file/d/1QoqcJkZBHsDporttTxaYWOM_ExSn7-Dz/view



#Observations

model.eval() is a kind of switch for some specific layers/parts of the model that behave differently during training and inference (evaluating) time. For example, Dropouts Layers, BatchNorm Layers etc. You need to turn off them during model evaluation, and .eval() will do it for you. In addition, the common practice for evaluating/validation is using torch.no_grad() in pair with model.eval() to turn off gradients computation:

# evaluate model:
model.eval()

with torch.no_grad():
    ...
    out_data = model(data)
    ...
BUT, don't forget to turn back to training mode after eval step:

# training step
...
model.train()
...

https://discuss.pytorch.org/t/loading-saved-models-gives-inconsistent-results-each-time/36312/24 

Also: 
could be due to dropout that is always active
