# Proxy-NCA
## Keras implementation of the paper, "No Fuss Distance Metric Learning using Proxies" by Movshovitz-Attias et al., 2017.

### Requirements
* Python 3
* Tensorflow
* Keras
* Numpy
* Pandas
* Matplotlib
#### </br>

### Dataset
#### Prepare train and validation data in the form of python dictionaries. The dictionary should be of the structure:
#### dict[class] = [datapoint0, datapoint1 ...] where datapoint0 is index 0 and so on.
#### Prepare train and validation csv files. The csv should be of the structure:
#### (class, index)
#### </br>

### Running the Code

#### ``` >> python train.py```
#### </br>

### References
#### Movshovitz-Attias, Y., Toshev, A., Leung, T. K., Ioffe, S., & Singh, S. (2017). No fuss distance metric learning using proxies. In Proceedings of the IEEE International Conference on Computer Vision (pp. 360-368).
