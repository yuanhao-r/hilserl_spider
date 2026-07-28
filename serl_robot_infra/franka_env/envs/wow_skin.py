# from anyskin import AnySkinProcess
import time
import numpy as np

class WowSkin():
    def __init__(self, port='/dev/ttyACM0'):
        self.baseline = np.zeros(15)
        self.sensor_stream = AnySkinProcess(num_mags=5, port=port)
        self.sensor_stream.start()
        time.sleep(1.0)  # 等待串口连接稳定
        
    def __del__(self):
        self.sensor_stream.pause_streaming()
        self.sensor_stream.join()
        
    def reset_baseline(self):
        """
        从实时数据流中采集一定数量的样本，取平均作为新的 baseline
        """
        baseline_data = self.sensor_stream.get_data(num_samples=5)
        baseline_data = np.array(baseline_data)[:, 1:]  # 跳过时间戳列
        baseline = np.mean(baseline_data, axis=0)
        self.baseline = baseline
        return baseline
    
    def get_force_data(self):
        sensor_data = self.sensor_stream.get_data(num_samples=1)[0][1:]
        force_data = sensor_data - self.baseline
        return force_data
    
if __name__ == "__main__":
    wowskin = WowSkin()
    hz = 1
    while True:
        wowskin.reset_baseline()
        print(wowskin.get_force_data())
        time.sleep(1.0 / hz)
    