# SDG Detector

SDG Detector is a software that checks the presence of SDG indicators in provided texts.

## Installation

Clone the repository and then install required libraries from ``requirements.txt``. The following command will install the latest version of the CoreNLP in the ``build`` directory. If you already have it somewhere else, just skip this step:
```bash
./configure
```

## Usage

First step is defining the list of URLs you want to check by adding them in ``urlsList.json``.  
After that you will be able to check if the text contains some of SDG references by executing this command if you installed CoreNLP by executing ``configure`` command:
```bash
./test
```
Alternatively, you can launch CoreNLP server from location where you installed it and then simply launch Python test script:
```bash
python3 test.py
