export HILSERL_ARM_BACKEND=tianji && \
export PYTHONPATH=/home/ubuntu/teleop_tianji/test_teleop/python:/home/ubuntu/teleop_tianji/test_teleop/demo:/home/ubuntu/teleop_tianji/hilserl_spider/serl_robot_infra:/home/ubuntu/teleop_tianji/hilserl_spider/serl_launcher:/home/ubuntu/teleop_tianji/hilserl_spider/examples:${PYTHONPATH} && \
export XLA_PYTHON_CLIENT_PREALLOCATE=true && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=.75 && \
python ../../train_rlpd.py \
    --exp_name=compounts_insertion \
    --learner \
    --checkpoint_path=first_run \
    "$@"
