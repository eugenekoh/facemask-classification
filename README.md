# facemask-classification

## Introduction 
This repository contains the code for my team's CZ4042 Project: Face Mask Recognition. The main training script is `train.py` and the training configurations can be modifed in `efficient_config.json` and `config.json`. The configuration files contain parameters such as the learning rate and location of dataset. The compiled dataset is stored in data folder where the main zip file is split into 100Mb sections. The `inference.py` script is also provided for live webcam evaluation. 

![example](assets/samples/white.gif)
## Quickstart

```bash
# setup virtual environment
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# train the model
python train.py --config config.json

# webcam evaluation on released efficient model, press ESC key to stop evalutation
python inference.py --model ./models/efficientnet
```

