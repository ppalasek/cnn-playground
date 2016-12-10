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

START=5
END=15
for i in $(seq ${START} ${END});
do
    echo ${i};
    python runCNN.py -i ${i} -r -m -a 'ccfff-ap'
    python runCNN.py -i ${i} -r -m -a 'ccfff-ap-d'
    python runCNN.py -i ${i} -r -m -a 'ccfff-mp'
    python runCNN.py -i ${i} -r -m -a 'ccfff-mp-d'
    python runCNN.py -i ${i} -r -m -a 'ccffsvm-ap'
    python runCNN.py -i ${i} -r -m -a 'ccffsvm-ap-d'
    python runCNN.py -i ${i} -r -m -a 'ccffsvm-mp'
    python runCNN.py -i ${i} -r -m -a 'ccffsvm-mp-d'
done
