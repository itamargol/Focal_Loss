## Focal_Loss Intro

Keras implementation for focal loss function.

Great mathematical solution for optimizing scenarios of unbalanced-classes.
Focal loss down-weights the well-classified examples (boosting-like concept). 
This has the net effect of putting more training emphasis on that data that is hard to classify. 

[Link to paper](https://arxiv.org/abs/1708.02002)

[Lighter Medium read](https://towardsdatascience.com/neural-networks-intuitions-3-focal-loss-for-dense-object-detection-paper-explanation-61bc0205114e)

![](https://github.com/itamargol/Focal_Loss/blob/master/focal_loss.png)

## How to use

``` python
from focal_loss import focal_loss

model.compile(loss=[focal_loss()], metrics=["accuracy"], optimizer=adam)

```     
