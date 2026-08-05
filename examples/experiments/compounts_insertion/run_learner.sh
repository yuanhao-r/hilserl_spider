export HILSERL_ARM_BACKEND=tianji && \
export PYTHONPATH=/home/ubuntu/teleop_tianji/test_teleop/python:/home/ubuntu/teleop_tianji/test_teleop/demo:/home/ubuntu/teleop_tianji/hilserl_spider/serl_robot_infra:/home/ubuntu/teleop_tianji/hilserl_spider/serl_launcher:/home/ubuntu/teleop_tianji/hilserl_spider/examples:${PYTHONPATH} && \
export XLA_PYTHON_CLIENT_PREALLOCATE=true && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.75 && \
python ../../train_rlpd.py \
    --exp_name=compounts_insertion \
    --checkpoint_path=0805train \
    --demo_path=./demo_data/compounts_insertion_20_demos_2026-08-04_14-32-18.pkl \
    --learner \
    --wandb \

