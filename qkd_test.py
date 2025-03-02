from measurement_Framework.coexistance import coexistance
import pandas as pd

if __name__ == '__main__':
    distances = [50, 80, 100, 150, 160]
    distances_lables = ['50km', '80km', '100km', '150km', '160km']
    df_skr = pd.DataFrame()
    df_qber = pd.DataFrame()
    for i in distances:
        obj = coexistance(1, 2, i)
        center_freq, slot_widths, signals, baud_rates = obj.channels_init(num_of_Q_ch=190)
        sprs_value_seo, spectral_info_out, power_out = obj.setup_channels(center_freq, slot_widths, signals, baud_rates)
        skr, qber, q_ch_wl = obj.get_sek_qber(sprs_value_seo)
        s1 = pd.Series(skr[::-1], index=q_ch_wl)
        q1 = pd.Series(qber[::-1], index=q_ch_wl)
        df_skr = pd.concat([df_skr, s1], axis=1)
        df_qber = pd.concat([df_qber, q1], axis=1)

    df_qber.columns = distances_lables
    df_skr.columns = distances_lables
    df_qber.to_csv('qber.csv')
    df_skr.to_csv('ker.csv')