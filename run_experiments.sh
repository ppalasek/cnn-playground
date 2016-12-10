#!/usr/bin/env bash

#
# Architectures
#
#    'ccfff-ap'
#    'ccfff-ap-d'
#    'ccfff-mp'
#    'ccfff-mp-d'
#    'ccffsvm-ap'
#    'ccffsvm-ap-d'
#    'ccffsvm-mp'
#    'ccffsvm-mp-d'
#

python runCNN.py -r -m -a 'ccfff-ap'
python runCNN.py -r -m -a 'ccfff-ap-d'
python runCNN.py -r -m -a 'ccfff-mp'
python runCNN.py -r -m -a 'ccfff-mp-d'
python runCNN.py -r -m -a 'ccffsvm-ap'
python runCNN.py -r -m -a 'ccffsvm-ap-d'
python runCNN.py -r -m -a 'ccffsvm-mp'
python runCNN.py -r -m -a 'ccffsvm-mp-d'
