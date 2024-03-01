# **Dual-Feedback Knowledge Retrieval for Task-Oriented Dialogue Systems**

## Requirements

### Retriever Pretraining

-   apex == 0.9.10dev
-   datasets == 2.0.0
-   deepspeed == 0.6.0
-   fairscale == 0.4.6
-   filelock == 3.6.0
-   packaging == 21.3
-   scikit-learn == 1.0.2
-   torch == 1.8.1
-   transformers == 4.17.0

### System Training

-   dataclasses == 0.8
-   filelock == 3.4.1
-   nltk == 3.6.7
-   packaging == 21.3
-   tensorboard == 2.9.1
-   torch == 1.8.1
-   transformers ==  3.0.2


## How to run

### Preparation

Please download from [this link](https://drive.google.com/file/d/1VIJOV7B0l3d2kVmR8Hnj9lsJmF7rLBfz/view?usp=sharing) for **all datasets** and **pretrained retriever models**. 


#### MWOZ

For the **dataset-level knowledge base** and **session-level knowledge base** of MWOZ, we provide **t5-base** and **t5-large** training scripts.

```
bash run_train.sh
```


#### SMD

For the **condensed knowledge base** of SMD, we provide **t5-base** and **t5-large** training scripts.

```
bash run_train_smd.sh
```

#### CamRest

For the **dataset-level knowledge base** and **session-level knowledge base** of CamRest, we provide **t5-base** and **t5-large** training scripts.

```
bash run_train_camrest.sh
```


