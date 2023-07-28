LS='CE'
MODEL='resnet50'

python attack.py --loss_function ${LS} --src_model ${MODEL}

python attack.py --MI --loss_function ${LS} --src_model ${MODEL}

python attack.py --DI --loss_function ${LS} --src_model ${MODEL}

python attack.py --TI --loss_function ${LS} --src_model ${MODEL}

python attack.py --SI_num 5 --loss_function ${LS} --src_model ${MODEL}

python attack.py --Admix_param 0.2 --loss_function ${LS} --src_model ${MODEL}

python attack.py --GSA --aug_num 15 --loss_function ${LS} --src_model ${MODEL}