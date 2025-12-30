# Start
```bash
docker start serl_container_1_5
xhost local:root
```

connect to arm:

| Arm | Host Address | Netmask | Gateway | Arm Address | Website Control |
| :---: | :---: | :---: | :---: | :---: | :---: |
| xarm | 192.168.1.100 | 255.255.255.0 | 192.168.1.1 | 192.168.1.232 | 192.168.1.232:18333 |
| realman | 192.168.124.100 | 255.255.255.0 | 192.168.124.1 | 192.168.124.18 | 192.168.124.18 |


## Data collect:
```bash
cd experiments/ram_insertion
python ../../record_demos.py --exp_name ram_insertion --successes_needed 20
```

## Training
run training:(delete first_run/ if you want to start a new training)
change run_actor.bash -replay_intervention_path for replay file
change run_learner.bash -demo_path for demo file

```bash
cd experiments/ram_insertion
bash run_actor.bash(press F8 to change to replay mode)
bash run_learner.bash
```

in run_actor.bash shell:
F6: pause
F8: replay mode
F9: self-run mode(default)
F4 or spacemouse: done and return True in self-run mode
esc or spacemouse: done and return False in self-run mode

## Eval
change run_actor.bash `--eval_checkpoint_step=CHECKPOINT_NUMBER_TO_EVAL` and `--eval_n_trajs=N_TIMES_TO_EVAL`

```bash
bash run_actor.bash
```

## Log:
if you have restarted the docker, use 'docker restart wandb-local-server' first
```bash
cd /home/wandb_data/wandb
wandb sync xxx
```

## Utils
1. `fisheye_test.py`: show the image from the fisheye camera.
    ```bash
    cd examples/utils
    python fisheye_test.py --index 0
    ```
    change the index to different camera. REMEMBER to change the index in config.py
    if you want to see the cropped image, change the `image_crop = None # lambda img: img[150:390, 500:820]` in fisheye_test.py

2. `xarm_control.py`: xarm api use, to get the status of the arm or control the arm
    ```bash
    cd examples/utils
    python xarm_control.py
    ```
    you can use the comment section of the code in `xarm_control.py` to achieve direct control using spacemouse.

3. `realman_spacemouse_control.py`: realman direct control using spacemouse.
    ```bash
    cd examples/utils
    python realman_spacemouse_control.py
    ```

4. `xarm_gripper_control.py`: control the gripper through the modbus(0 for release, 1 for close)
    ```bash
    python xarm_gripper_control.py --cmd 0
    ```