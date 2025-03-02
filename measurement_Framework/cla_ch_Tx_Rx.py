import numpy as np
import os, json

def load_conf(file):
    with open(conf_file(file), 'r') as f:
        return json.load(f)

def conf_file(file):
    return os.path.join('measurement_Framework', 'config',file)

class cla_ch_Tx_Rx:
    def __init__(self, dbm2watt_obj, tx, rx, fiber_length):
        self.config = load_conf('classical_channel.json')
        self.Tx = tx
        self.Rx = rx
        self.fiber_length = fiber_length
        self._channel_power = self.config['channel_power']
        self._classical_channel_bandwidth = self.config['classical_channel_bandwidth']
        self._guard_bandwidth = self.config['guard_bandwidth']
        self._guard_power = self.config['guard_power']
        self._baud_rate_classical_channel = self.config['baud_rate_classical_channel']
        self.roll_off = self.config['roll_off']
        self.dbm2watt = dbm2watt_obj
        self.channels = {}


    def channels_init(self, freq_band, guard=True):
        if guard:
            BW = self._guard_bandwidth
            P = self._guard_power
        else:
            BW = self._classical_channel_bandwidth
            P = self._channel_power
        bandwidth = (freq_band[1] - freq_band[0])
        num_slots = int(bandwidth / BW)
        start_freq = freq_band[0] + BW / 2
        center_freq = np.zeros(num_slots)
        for i in range(num_slots):
            f_i = start_freq + (i * BW)
            center_freq[i] = f_i
        slot_width = np.array([BW for _ in range(num_slots)])
        baud_rate_array = np.array([self._baud_rate_classical_channel for _ in range(num_slots)])
        signal = self.dbm2watt(np.array([P for _ in range(num_slots)]))
        return bandwidth, num_slots, center_freq, slot_width, signal, baud_rate_array

    def establish_channels(self, band, freq):
        num_of_ch = int(np.floor((freq[1] - freq[0]) / self._classical_channel_bandwidth))
        step = (freq[1] - freq[0]) / num_of_ch
        already_satup = len(self.channels)
        for i in range(num_of_ch):
            self.channels[f'{band}_{i+already_satup}'] = (freq[0] + (i * step), freq[0] + ((i + 1) * step))



