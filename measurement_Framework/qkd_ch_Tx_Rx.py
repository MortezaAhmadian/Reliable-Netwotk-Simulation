import json, os, math
import numpy as np
from gnpy.tools.json_io import load_json
from scipy.interpolate import interp1d


def wavelength_freq_conversion(x):
    return 3e8 / x

def load_conf(file):
    with open(conf_file(file), 'r') as f:
        return json.load(f)

def conf_file(file):
    return os.path.join('measurement_Framework', 'config',file)

def sprs_cross_section_fun(input_wave):
    SpRS_cross_section = load_json(conf_file('cross_section.json'))
    wave = [x * 1e-9 for x in SpRS_cross_section['x']]
    coef = [x * 1e-9 for x in SpRS_cross_section['y']]
    func = interp1d(wave, coef, kind='linear')
    return func(input_wave)


def _sprs_power_func(p_ch, L, alpha, Δλ, λ_d, λ_q):
    λ_δ = 1 / ((1 / (1550 * 1e-9)) + (1 / λ_q) - (1 / λ_d))
    ρ_1550_δ = sprs_cross_section_fun(λ_δ)
    # print(λ_δ,ρ_1550_δ)
    ρ = ((λ_δ / λ_q) ** 4) * ρ_1550_δ
    p_sprs = p_ch * L * np.exp(-1 * alpha * L) * ρ * Δλ
    return p_sprs

def H_shanon(x):
    if (x == 0):
        raise ValueError("Input of function H_shanon cannot be zero")
    if (x == 1):
        raise ValueError("Input of function H_shanon cannot be one")

    return (-x * np.log2(x)) - ((1 - x) * np.log2(1 - x))


class qkd_ch_Tx_Rx:
    def __init__(self, dbm2watt_obj, qtx, qrx, fiber_length):
        self.config = load_conf('quantum_channel.json')
        self.QTx = qtx
        self.QRx = qrx
        self.fiber_length = fiber_length
        self._quantum_bandwidth = self.config['quantum_bandwidth']
        self._quantum_power = self.config['quantum_power']
        self._baud_rate = self.config['baud_rate']
        self._detectors_quantum_efficiency = self.config['detectors_quantum_efficiency']
        self._phase_distortion = self.config['phase_distortion']
        self._photon_per_pulse = self.config['photon_per_pulse']
        self._dark_count_rate = self.config['dark_count_rate']
        self._pulse_repetition_time = self.config['pulse_repetition_time']
        self._gate_interval_time = self.config['gate_interval_time']
        self._error_correction_inefficiency = self.config['error_correction_inefficiency']
        self.dbm2watt = dbm2watt_obj
        self.channels_signal = None
        self.channels_baud_rate = None
        self.channels_bandwidth = None
        self.channels_freq = None
        self.channels = None
        self.num_of_channels = self.config['num_of_ch']


    def channels_init(self):
        self.channels = [((i*0.08) + 1260) * 1e-9 for i in range(self.num_of_channels)]
        self.channels_freq = ([wavelength_freq_conversion(self.channels[i]) for i in range(len(self.channels))])
        self.channels_bandwidth = np.array([self._quantum_bandwidth] * len(self.channels))
        self.channels_signal = np.array([self.dbm2watt(self._quantum_power)] * len(self.channels))
        self.channels_baud_rate = np.array([self._baud_rate] * len(self.channels))


    def return_quantum_channels(self):
        return self.channels_freq, self.channels_bandwidth, self.channels_signal, self.channels_baud_rate

    def calculate_sprs_seo(self, num_C_ch, num_guard_ch, signal_input_unchanged, si, fiber):
        sprs_value_seo = []
        for q in range(self.num_of_channels):
            sprs_pow = 0
            λ_q = wavelength_freq_conversion(si.frequency[(num_C_ch + num_guard_ch) + q])
            Δλ = ((wavelength_freq_conversion(
                (si.frequency[(num_C_ch + num_guard_ch) + q]) - self._quantum_bandwidth)) - λ_q) / 1e-9
            for c in range(num_C_ch + num_guard_ch):
                alpha_C = ((fiber.params.loss_coef[c]) * 1000) / 4.343
                f_i = si.frequency[c]
                λ_d = wavelength_freq_conversion(f_i)  # find wavelengh of classic ch.
                P_i = signal_input_unchanged.signal[c]
                sprs_pow += _sprs_power_func(P_i, self.fiber_length, alpha_C, Δλ, λ_d, λ_q)
            sprs_value_seo.append(sprs_pow)
        return np.array(sprs_value_seo)

    def num_photon(self, pow, λ):
        """
        h is planck const. and c is light speed.
        h =  6.62607015*(10**(-34)).
        """
        c = 3e8
        h = 6.022e-34
        return pow * λ * self._gate_interval_time * self._detectors_quantum_efficiency * (1 / (2 * h * c))

    def skr_qber(self, alpha_q_array_seo, nli_Q_photon_seo, sprs_Q_photon_seo):
        skr = []
        qber = []
        for i in range(len(alpha_q_array_seo)):
            P_b = self._dark_count_rate + nli_Q_photon_seo[i] + sprs_Q_photon_seo[i]
            Y_0 = self._error_cal(alpha_q_array_seo[i], self.fiber_length, 0, P_b)[1]
            Y_1 = self._error_cal(alpha_q_array_seo[i], self.fiber_length, 1, P_b)[1]
            η = self._detectors_quantum_efficiency * np.exp(-alpha_q_array_seo[i] * self.fiber_length)

            Q_μ = (1 - ((1 - Y_0) * (math.exp(-η * self._photon_per_pulse))))
            if Q_μ == 0:  # in mathematical cal. it was set to zero
                Q_μ = Y_0
            # quantum bit error rate
            E_μ = (((Y_0 / 2) + self._phase_distortion * (
                        1 - np.exp(-η * self._photon_per_pulse))) / Q_μ)
            qber.append(E_μ)
            Q_1 = Y_1 * np.exp(-self._photon_per_pulse) * self._photon_per_pulse
            e_1 = (self._phase_distortion * η + 0.5 * P_b) / Y_1
            rate = ((-self._error_correction_inefficiency * Q_μ * H_shanon(E_μ) + (Q_1 * (1 - H_shanon(e_1)))) /
                    self._pulse_repetition_time)
            skr.append(max(0, rate))
        return skr, qber


    def _error_cal(self, alpha, L, n, P_b):
        η = self._detectors_quantum_efficiency * np.exp(-alpha * L)
        η_n = 1 - ((1 - η) ** n)
        if n == 0:
            Y_n = P_b
        else:
            Y_n = η_n + (P_b) - (η_n * P_b)
        return η_n, Y_n





        

