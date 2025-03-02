from gnpy.tools.json_io import load_json
from gnpy.core.parameters import SimParams
from gnpy.core.utils import dbm2watt
import os, sys
import numpy as np
from .cla_ch_Tx_Rx import cla_ch_Tx_Rx
from .qkd_ch_Tx_Rx import qkd_ch_Tx_Rx
from scipy.interpolate import interp1d
from gnpy.core.info import create_arbitrary_spectral_information
from gnpy.core.utils import lin2db
from gnpy.core.elements import Fiber


def wavelength_freq_conversion(x):
    return 3e8 / x

def conf_file(file):
    return os.path.join('measurement_Framework', 'config', file)

def alph_func(input_wave):
    alpha = load_json(conf_file('alpha.json'))
    wave = [x * 1e-9 for x in alpha['x']]
    func = interp1d(wave, alpha['y'], kind='linear')
    return func(input_wave)


class coexistance:
    def __init__(self, tx, rx, fiber_length):
        SimParams.set_params(load_json(conf_file('core_simulation_params.json')))
        self.fiber_length = fiber_length
        self.freq_bands = load_json(conf_file('frequency_bands.json'))
        self.classical_channel = cla_ch_Tx_Rx(dbm2watt, tx, rx, fiber_length)
        self.qkd_channels = qkd_ch_Tx_Rx(dbm2watt, tx, rx, fiber_length)
        self.channel_S = None
        self.guard_band_SC = None
        self.channel_C = None
        self.guard_band_CL = None
        self.channel_L = None
        self.channels_Q = None
        self.fiber = None
        self.si = None
        self.si_unchanged = None


    def channels_init(self):
        self.channel_S = self.classical_channel.channels_init(self.freq_bands['S_BAND'], guard=False)
        self.guard_band_SC = self.classical_channel.channels_init(self.freq_bands['SC_GUARD'], guard=True)
        self.channel_C = self.classical_channel.channels_init(self.freq_bands['C_BAND'], guard=False)
        self.guard_band_CL = self.classical_channel.channels_init(self.freq_bands['CL_GUARD'], guard=True)
        self.channel_L = self.classical_channel.channels_init(self.freq_bands['L_BAND'], guard=False)
        self.qkd_channels.channels_init()
        self.channels_Q = self.qkd_channels.return_quantum_channels()
        return self._combine_channels()

    def _combine_channels(self):
        lst = []
        for i in range(4):
            lst.append([*self.channel_L[i + 2], *self.guard_band_CL[i + 2],
                        *self.channel_C[i + 2], *self.guard_band_SC[i + 2],
                        *self.channel_S[i + 2], *self.channels_Q[i]])
        return lst

    def setup_channels(self, center_freq, slot_widths, signals, baud_rates):
        roll_off = np.array([self.classical_channel.roll_off for _ in range(len(center_freq))])
        self.si_unchanged = create_arbitrary_spectral_information(frequency=np.array(center_freq),
                                                                      slot_width=np.array(slot_widths),
                                                                      signal=np.array(signals),
                                                                      baud_rate=np.array(baud_rates),
                                                                      roll_off=roll_off,
                                                                      tx_osnr=np.array([18] * len(signals)))
        self.si = create_arbitrary_spectral_information(frequency=np.array(center_freq),
                                                   slot_width=np.array(slot_widths),
                                                   signal=np.array(signals), baud_rate=np.array(baud_rates),
                                                   roll_off=roll_off, tx_osnr=np.array([18] * len(signals)))
        spectral_info_out, power_out = self.launch_signal_in_fiber()
        sprs_value_seo = self.qkd_channels.calculate_sprs_seo(self.channel_C[1]+self.channel_L[1]+self.channel_S[1],
                                                              self.guard_band_CL[1]+self.guard_band_SC[1],
                                                              self.si_unchanged, self.si, self.fiber)
        return sprs_value_seo, spectral_info_out, power_out


    def launch_signal_in_fiber(self):
        center_freq_alpha = self.si.frequency
        loss_coef = np.zeros(self.si.number_of_channels)
        for f in range(self.si.number_of_channels):
            freq = center_freq_alpha[f]
            alpha = alph_func(wavelength_freq_conversion(freq))
            loss_coef[f] = alpha
        loss_coef = loss_coef.tolist()
        loss_coef_dict = {'value': loss_coef, 'frequency': center_freq_alpha.tolist()}
        ref_frequency = self.si.frequency[int((self.si.number_of_channels / 2))]
        data = load_json(conf_file('scl_alpha.json'))
        data['length'] = self.fiber_length
        data['ref_frequency'] = ref_frequency
        data['loss_coef'] = loss_coef_dict
        self.fiber = Fiber(params=data, uid="Span1")
        spectral_info_out = self.fiber(self.si)
        power_out = lin2db(spectral_info_out.signal * 1e3)
        return spectral_info_out, power_out

    def get_sek_qber(self, sprs_seo):
        """
        first gets nli of the quantum channels.
        second, gets fiber attenuation factor.
        then gets number of photons which are introduced by nli and sprs.
        then the skr and qber are computed.
        """
        num_non_q_ch = (self.channel_C[1] + self.channel_L[1] + self.guard_band_CL[1] +
                        self.channel_S[1] + self.guard_band_SC[1])
        nli_q_seo = self.si.nli[num_non_q_ch:]
        λ_q_seo = wavelength_freq_conversion(self.si.frequency[num_non_q_ch:])
        alpha_q_seo = ((self.fiber.params.loss_coef[num_non_q_ch:]) * 1000) / 4.343
        nli_q_photon_seo = [self.qkd_channels.num_photon(nli_q_seo[i], λ_q_seo[i]) for i in range(len(nli_q_seo))]
        sprs_q_photon_seo = [self.qkd_channels.num_photon(sprs_seo[i], λ_q_seo[i]) for i in range(len(sprs_seo))]
        skr, qber = self.qkd_channels.skr_qber(alpha_q_seo, nli_q_photon_seo, sprs_q_photon_seo)
        return skr, qber, self.qkd_channels.channels



