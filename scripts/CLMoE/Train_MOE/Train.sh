bash ./scripts/CLMoE/Train_MOE/1_recognition.sh
python statistic.py --task recognition
bash ./scripts/CLMoE/Train_MOE/2_location.sh
python statistic.py --task location
python params.py --task1 recognition --task2 location
bash ./scripts/CLMoE/Train_MOE/3_judge.sh
python statistic.py --task judge
python params.py --task1 location --task2 judge
bash ./scripts/CLMoE/Train_MOE/4_commonsense.sh
python statistic.py --task commonsense
python params.py --task1 judge --task2 commonsense
bash ./scripts/CLMoE/Train_MOE/5_count.sh
python statistic.py --task count
python params.py --task1 commonsense --task2 count
bash ./scripts/CLMoE/Train_MOE/6_action.sh
python statistic.py --task action
python params.py --task1 count --task2 action
bash ./scripts/CLMoE/Train_MOE/7_color.sh
python statistic.py --task color
python params.py --task1 action --task2 color
bash ./scripts/CLMoE/Train_MOE/8_type.sh
python statistic.py --task type
python params.py --task1 color --task2 type
bash ./scripts/CLMoE/Train_MOE/9_subcategory.sh
python statistic.py --task subcategory
python params.py --task1 type --task2 subcategory
bash ./scripts/CLMoE/Train_MOE/10_causal.sh
python statistic.py --task causal 
python params.py --task1 subcategory --task2 causal