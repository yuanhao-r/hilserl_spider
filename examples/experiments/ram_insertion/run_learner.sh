export HILSERL_ARM_BACKEND=tianji && \
export PYTHONPATH=/home/ubuntu/teleop_tianji/test_teleop/python:/home/ubuntu/teleop_tianji/test_teleop/demo:/home/ubuntu/teleop_tianji/hilserl_spider/serl_robot_infra:/home/ubuntu/teleop_tianji/hilserl_spider/serl_launcher:/home/ubuntu/teleop_tianji/hilserl_spider/examples:${PYTHONPATH} && \
export XLA_PYTHON_CLIENT_PREALLOCATE=true && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.6 && \
python ../../train_rlpd.py "$@" \
    --exp_name=ram_insertion \
    --checkpoint_path=first_run_260529 \
    --demo_path=./demo_data/ram_insertion_20_demos_2026-05-29_14-15-40.pkl \
    --learner \
    --wandb \

# ./demo_data/ram_insertion_20_demos_2026-05-18_14-58-17.pkl
#./demo_data/ram_insertion_20_demos_2026-05-11_13-43-37.pkl \

