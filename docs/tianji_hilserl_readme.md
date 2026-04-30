# 天机机械臂HIL-SERL强化学习项目

# Start
When restart the computer(with GPU4090) or restart the robot, use command bellow to reconnect robotarm.
```bash
cd /home/ubuntu/teleop_tianji/test_teleop 
# 网卡ip设置到10.10.12网段（测试用10.10.12.172）
bash add_route.sh

#可选：运行以下命令，能看到 10.10.13.0      _gateway        255.255.255.0  就是设置成功了
sudo route
ping 10.10.13.10 #能ping通代表中转成功，可以通过4090主机连接orin从而连接机械臂
```

所有代码均需要在hilserl_tianji的conda环境中运行

`http://192.168.5.103:3001/control`网址是这台机器人的web端控制界面，可以实现机器人的控制，例如下腰等

## 上位机软件调试机械臂
```bash
cd /home/ubuntu/aaa_tianji_robotarm/marvin_app/MARVIN_APP_UBUNTU_WINDOWS
python UI_FX2.py 
```
ip 10.10.13.10， 连接机器人后
（阻抗模式）点击：清错->复位->关节阻抗->关节拖动，然后按住机械臂末端圆钮，即可手动拖动机械臂到任意可到达位置
（位置模式）点击：清错->复位->关节跟随->加点（第三大行处 1#加点 / 2#加点）->运行（第三大行处 1#运行 / 2#运行），即可通过设置7个关节角的方式控制机械臂到达指定位姿
### 上位机端和代码控制不能同时连机械臂，如果要使用代码控制则需要在上位机端断开连接！

# Data collect:
```bash
cd experiments/ram_insertion
export HILSERL_ARM_BACKEND=tianji
python ../../record_demos.py --exp_name ram_insertion --successes_needed 20
```

# Training
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
for example:
```bash
./run_actor.sh --eval_checkpoint_step=34000 --eval_n_trajs=20
```

```bash
bash run_actor.bash
```
# Utils(hilserl_spider/examples/utils)
1. `fisheye_test.py`: show the image from the fisheye camera.
    ```bash
    cd examples/utils
    python fisheye_test.py --index 0
    ```
    change the index to different camera. REMEMBER to change the index in config.py
    if you want to see the cropped image, change the `image_crop = None # lambda img: img[150:390, 500:820]` in fisheye_test.py

2.  `gripper_control.py`: 检测夹爪能否正常开闭

3.  `marvin_joint_control_with_feedback.py`: 控制机械臂到指定位姿，`MONITOR_ONLY = True`表示不控制(机械臂不动)，用于读取当前机械臂状态信息（关节角，末端位姿等），`MONITOR_ONLY = False`表示设置位姿并运动到目标位姿，真实运动。
可通过修改`TARGET_JOINTS`方便地通过关节角度控制机械臂。
注意改IP和左臂右臂。

4.  `marvin_gripper_control.py`:控制钧舵夹爪开闭，夹爪第一次上电需要夹爪自检（开闭检测夹爪行程），此后不需要
```bash
#默认：会先激活再执行
python marvin_gripper_control.py --cmd 0
python marvin_gripper_control.py --cmd 1
# 跳过激活，直接张开或者直接闭合（确认夹爪已激活时再用）
python marvin_gripper_control.py --cmd 0 --skip-activate
python marvin_gripper_control.py --cmd 1 --skip-activate
```
注意改IP和左臂右臂。


5.  `marvin_cartesian_move.py`: 可通过修改`TARGET_XYZABC`方便地通过末端位姿控制机械臂，可通过直接对xyz三轴值的加减处理方便地调节末端位姿

对于右臂来说，正前方是x轴正方向，上方是y轴正方向，右边是z轴正方向，即机械臂关节角全0（竖直）状态时是沿着z轴正方向伸直的


# 启动方法
## 准备阶段
通过`marvin_cartesian_move.py`控制末端xyz移动到抓取点、抓取点的上方点（y+100）、RESET点（上方点的基础上x-10），在三个点分别运行`marvin_joint_control_with_feedback.py`(`MONITOR_ONLY = False`)，读出每个点的关节角，分别填入`hilserl_spider/examples/experiments/ram_insertion/config.py`中的`TARGET_JOINTS(GRASP_JOINTS)`,`TOP_JOINTS`,`RESET_JOINTS`
## 人类专家数据收集
运行
```bash
python ../../record_demos.py --exp_nameram_insertion --successes_needed 20
```
然后使用spacemouse操控，左键表示成功，右键表示失败

## 训练
```bash
cd teleop_tianji/hilserl_spider/examples/experiments/ram_insertion
bash run_actor.sh
#另开一终端
bash run_learner.sh
```
开始训练，spacemouse操控介入，左键表示成功，右键表示失败。
根据需要确定是否保留`hilserl_spider/examples/experiments/ram_insertion/first_run`目录