export HILSERL_ARM_BACKEND=tianji && \
export PYTHONPATH=/home/ubuntu/teleop_tianji/test_teleop/python:/home/ubuntu/teleop_tianji/test_teleop/demo:/home/ubuntu/teleop_tianji/hilserl_spider/serl_robot_infra:/home/ubuntu/teleop_tianji/hilserl_spider/serl_launcher:/home/ubuntu/teleop_tianji/hilserl_spider/examples:${PYTHONPATH} && \
export XLA_PYTHON_CLIENT_PREALLOCATE=false && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.3 && \
python ../../train_rlpd.py "$@" \
    --exp_name=ram_insertion \
    --checkpoint_path=first_run \
    --demo_path=./demo_data/ram_insertion_20_demos_2026-04-28_14-28-43.pkl \
    --learner \
    --wandb \
