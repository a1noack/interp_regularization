# Interpretation regularization
A novel neural network gradient regularization scheme for adversarial robustness and interpretability. It works by encouraging the gradients of the neural network logits to (1) have small magnitudes, like Jacobian regularization, and (2) align with supplied attribution maps. Our findings indicate that the more the supplied attribution maps highlight features that are robust, the more robust the network being trained with our method becomes.

Read [our published paper](https://link.springer.com/article/10.1007/s42979-020-00390-x) for more information on the method.

### Installation
Run `$ python setup.py install` to install the ir_utils package.
