# Principal Component Networks

Original (messy) source code for 
[Principal Component Networks: Parameter Reduction Early in Training](https://arxiv.org/abs/2006.13347).

Code for the paper can be found in the `pcn` folder. The beginnings of a
refactor meant to clean up the code can be found in `pcns_v2`.

### V2 Status
Forward (input) transformation for CIFAR complete. ImageNet code & backward (output) transformation not started. Additional
adversarial training functionality has been added.

### Installation Note
The imports are with respect to the base directory. This may cause errors depending on what directory the code is run
from. To avoid this simply:
```
export PYTHONPATH=$PYTHONPATH:{path to base directory}
```
or in your Dockerfile:
```
ENV PYTHONPATH "${PYTHONPATH}:{path to base directory}"
```

### Base scripts

#### V2
```
python src/pcns_v2/experiment_scripts/vgg16A_on_cifar.py
python src/pcns_v2/experiment_scripts/resnet_on_cifar.py

python src/pcns_v2/adversarial/adv_resnet_on_cifar.py
```

#### V1
```
python src/pcns/vgg16.py
python src/pcns/resnet/cifar10.py
```



