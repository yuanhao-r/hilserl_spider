import sys
import os
import time
import threading
import logging
import argparse

# 1. 获取当前文件所在目录: .../hilserl_spider/examples/utils
current_dir = os.path.dirname(os.path.abspath(__file__))

# 2. 向上跳三级，到达共同根目录 /home/ubuntu/teleop_tianji
# 第一级回退到 examples, 第二级回退到 hilserl_spider, 第三级回退到 teleop_tianji
root_dir = os.path.abspath(os.path.join(current_dir, "../../.."))

# 3. 拼接出 SDK 所在的 demo 目录: /home/ubuntu/teleop_tianji/test_teleop/demo
sdk_parent_path = os.path.join(root_dir, "test_teleop", "demo")

# 4. 将该路径加入搜索路径
if sdk_parent_path not in sys.path:
    sys.path.insert(0, sdk_parent_path)

# 引入天机机器人SDK
try:
    from SDK_PYTHON.fx_robot import Marvin_Robot
    # print(f"成功加载 SDK，路径来自于: {sdk_parent_path}")
except ImportError:
    print(f"Error: 找不到 SDK_PYTHON。")
    print(f"请检查路径是否存在: {sdk_parent_path}")
    print(f"当前 sys.path 为: {sys.path}")
    sys.exit(1)

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('gripper')

class GripperController:
    def __init__(self, robot_ip, robot_id='B', com_port=2, slave_id=9):
        """
        初始化夹爪控制器
        :param robot_ip: 机械臂IP地址
        :param robot_id: 机械臂ID ('A' 或 'B')
        :param com_port: 485端口号 (2 代表 com1/末端485)
        :param slave_id: 夹爪Modbus站号 (默认 9)
        """
        self.robot = Marvin_Robot()
        self.ip = robot_ip
        self.robot_id = robot_id
        self.com_port = com_port
        self.slave_id = slave_id
        self.is_connected = False
        self.is_activated = False
        self._stop_event = threading.Event()

    def connect(self):
        """连接机械臂并开启接收线程"""
        logger.info(f"正在连接机械臂 IP: {self.ip} ...")
        init = self.robot.connect(self.ip)
        if init != 1:
            logger.error("机械臂连接失败，请检查网络或是否被占用。")
            return False

        # 初始化设置：清除错误，清除缓存
        self.robot.clear_set()
        self.robot.clear_error(self.robot_id)
        self.robot.send_cmd()
        time.sleep(0.5)
        self.robot.clear_485_cache(self.robot_id)
        
        # 开启接收线程 (防止缓存堆积，虽然我们这里做盲发控制，但保持读取是好习惯)
        self.read_thread = threading.Thread(target=self._read_loop, daemon=True)
        self.read_thread.start()
        
        self.is_connected = True
        self.is_activated = False
        logger.info("机械臂连接成功，485通信已就绪。")
        return True

    def _read_loop(self):
        """后台线程：持续读取485数据，防止缓冲区溢出"""
        while not self._stop_event.is_set():
            try:
                # 读取数据但不处理，主要为了清空缓存
                # 如果需要获取“抓取到位”信号，需解析这里的返回数据
                tag, _ = self.robot.get_485_data(self.robot_id, self.com_port)
                if tag < 1:
                    time.sleep(0.01)
            except Exception:
                pass

    def _calc_crc16(self, data: bytearray) -> int:
        """计算Modbus RTU CRC16校验码"""
        crc = 0xFFFF
        for pos in data:
            crc ^= pos
            for i in range(8):
                if (crc & 1) != 0:
                    crc >>= 1
                    crc ^= 0xA001
                else:
                    crc >>= 1
        return crc

    def _send_modbus_cmd(self, payload: list):
        """
        构建并发送Modbus指令
        :param payload: 指令字节列表 (不含CRC)
        """
        if not self.is_connected:
            logger.error("未连接机械臂")
            return

        # 1. 准备数据和计算CRC
        data_bytes = bytearray(payload)
        crc = self._calc_crc16(data_bytes)
        
        # 2. 添加CRC (低字节在前，高字节在后)
        data_bytes.append(crc & 0xFF)
        data_bytes.append((crc >> 8) & 0xFF)

        # 3. 转换为机械臂SDK需要的HEX字符串格式 (例如 "09 10 03 E8 ...")
        hex_str = " ".join([f"{b:02X}" for b in data_bytes])
        byte_len = len(data_bytes)
        
        logger.debug(f"发送指令: {hex_str}")
        
        # 4. 通过机械臂透传发送
        success, _ = self.robot.set_485_data(
            self.robot_id, 
            hex_str, 
            byte_len,
            self.com_port
        )
        
        if not success:
            logger.error("指令发送失败")
        
        # 适当延时，给夹爪反应时间
        time.sleep(0.05)

    def activate(self):
        """
        激活/使能夹爪 (参考PDF 7.1)
        必须先写0重置，再写1使能
        """
        logger.info("正在激活夹爪...")
        
        # 步骤1: 清除使能 (Write 0 to register 0x03E8)
        # 09 10 03 E8 00 01 02 00 00
        cmd_reset = [
            self.slave_id, 0x10,    # ID, 功能码(写多个)
            0x03, 0xE8,             # 起始地址 1000
            0x00, 0x01,             # 寄存器数量 1
            0x02,                   # 字节数
            0x00, 0x00              # 数据: 0
        ]
        self._send_modbus_cmd(cmd_reset)
        time.sleep(0.5)

        # 步骤2: 设置使能 (Write 1 to register 0x03E8)
        # 09 10 03 E8 00 01 02 00 01
        cmd_enable = [
            self.slave_id, 0x10,
            0x03, 0xE8,
            0x00, 0x01,
            0x02,
            0x00, 0x01               # 数据: 1 (rACT=1)
        ]
        self._send_modbus_cmd(cmd_enable)
        self.is_activated = True
        logger.info("激活指令已发送，等待夹爪初始化(约2秒)...")
        time.sleep(2.0) # 等待夹爪完成自检动作

    def move(self, position, speed=255, force=255, auto_activate=True):
        """
        控制夹爪移动 (参考PDF 7.1 含参数控制)
        :param position: 位置 0-255 (0=全开, 255=全闭)
        :param speed: 速度 1-255
        :param force: 力矩 1-255
        :param auto_activate: 未激活时是否自动激活
        """
        if not self.is_activated:
            if auto_activate:
                logger.info("夹爪未激活，先执行激活...")
                self.activate()
            else:
                logger.warning("夹爪未激活，按参数跳过激活，直接发送移动指令")

        # 根据协议(PDF Page 32/37/39)，我们需要连续写入3个寄存器：
        # Reg 0x03E8 (控制): 0x0009 (rACT=1, rGTO=1) -> 保持激活并GoTo目标
        # Reg 0x03E9 (位置): HighByte=Pos, LowByte=Reserved(0)
        # Reg 0x03EA (力/速): HighByte=Force, LowByte=Speed
        
        pos = int(position) & 0xFF
        spd = int(speed) & 0xFF
        frc = int(force) & 0xFF

        logger.info(f"控制夹爪: 位置={pos}, 速度={spd}, 力度={frc}")

        payload = [
            self.slave_id, 0x10,    # ID, 写多个
            0x03, 0xE8,             # 起始地址 0x03E8
            0x00, 0x03,             # 寄存器数量 3
            0x06,                   # 字节数 6
            # Reg 0x03E8: 00 09 (rACT=1, rGTO=1)
            0x00, 0x09,             
            # Reg 0x03E9: Position, Reserved
            pos, 0x00,              
            # Reg 0x03EA: Force, Speed
            frc, spd                
        ]
        
        self._send_modbus_cmd(payload)

    def close_gripper(self):
        """完全闭合"""
        self.move(position=255, speed=255, force=255)

    def open_gripper(self):
        """完全张开"""
        self.move(position=0, speed=255, force=255)

    def disconnect(self):
        """断开连接"""
        self._stop_event.set()
        if hasattr(self, "read_thread") and self.read_thread.is_alive():
            self.read_thread.join(timeout=1.0)
        self.robot.release_robot()
        logger.info("已断开连接")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Marvin 夹爪控制脚本：支持命令行开/闭夹爪"
    )
    parser.add_argument(
        "--ip",
        default="10.10.13.10",
        help="机械臂 IP 地址，默认: 10.10.13.10",
    )
    parser.add_argument(
        "--arm",
        default="B",
        choices=["A", "B"],
        help="机械臂 ID，A 或 B，默认: B",
    )
    parser.add_argument(
        "--com-port",
        type=int,
        default=2,
        help="485 端口号，默认: 2 (com1/末端485)",
    )
    parser.add_argument(
        "--slave-id",
        type=int,
        default=9,
        help="夹爪 Modbus 站号，默认: 9",
    )
    parser.add_argument(
        "--cmd",
        "---cmd",
        type=int,
        choices=[0, 1],
        default=None,
        help="夹爪指令：0=打开，1=闭合。未提供时运行原测试流程。",
    )
    parser.add_argument(
        "--speed",
        type=int,
        default=255,
        help="夹爪速度 1-255，默认: 255",
    )
    parser.add_argument(
        "--force",
        type=int,
        default=255,
        help="夹爪力度 1-255，默认: 255",
    )
    parser.add_argument(
        "--wait",
        type=float,
        default=1.0,
        help="指令发送后等待时间（秒），默认: 1.0",
    )
    parser.add_argument(
        "--skip-activate",
        action="store_true",
        help="跳过激活步骤（仅在已确认夹爪处于激活状态时使用）",
    )
    return parser.parse_args()


def run_cmd_mode(gripper, args):
    """命令行单指令模式：--cmd 0 打开，--cmd 1 闭合"""
    if args.skip_activate:
        logger.info("按参数跳过激活步骤")
    else:
        gripper.activate()

    if args.cmd == 0:
        logger.info("执行命令: 打开夹爪 (--cmd 0)")
        gripper.move(
            position=0,
            speed=args.speed,
            force=args.force,
            auto_activate=not args.skip_activate
        )
    else:
        logger.info("执行命令: 闭合夹爪 (--cmd 1)")
        gripper.move(
            position=255,
            speed=args.speed,
            force=args.force,
            auto_activate=not args.skip_activate
        )
    time.sleep(max(0.0, args.wait))
    logger.info("命令执行完成")

# --- 主程序测试示例 ---
if __name__ == "__main__":
    args = parse_args()
    gripper = GripperController(
        robot_ip=args.ip,
        robot_id=args.arm,
        com_port=args.com_port,
        slave_id=args.slave_id
    )

    if gripper.connect():
        try:
            if args.cmd is not None:
                run_cmd_mode(gripper, args)
            else:
                # 原有演示流程（未指定 --cmd 时）
                gripper.activate()

                print("正在闭合夹爪...")
                gripper.close_gripper()
                time.sleep(2)

                print("正在张开夹爪...")
                gripper.open_gripper()
                time.sleep(2)

                print("移动到中间位置...")
                gripper.move(position=128, speed=100, force=50)
                time.sleep(2)

                print("测试完成。")

        except KeyboardInterrupt:
            print("用户中断")
        finally:
            gripper.disconnect()
