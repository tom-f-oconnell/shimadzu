
import os

import pandas as pd

import shimadzu


# TODO also make a function to tell whether a given m/z is saturated, and test
# that?

def test_get_saturated_peaks():
    """
    """
    sample_data = \
        shimadzu.read('example_data/181026_mango_N3_split2_onlyTICMIC.txt')
    data = sample_data['181026_mango_N3_split2']

    cdf_directory = 'example_data'
    qgd_filename = data['header'].at['data_file_name',1].split('\\')[-1]
    cdf_export = os.path.join(cdf_directory, qgd_filename[:-4] + '.CDF')

    peak_table = data['peak_table']

    sat_peaks = shimadzu.get_saturated_peaks(cdf_export, peak_table)

    # All unlabeled masses in any labelled samples are assumed to be
    # unsaturated, so it is important to label all saturated peaks in a given
    # sample.
    # Columns: sample_name, start_(scan/rettime?), end_scan, m/z
    truth = pd.read_csv('../example_data/saturation_truth.csv')

    true_sat_peaks = []
    for peak_id, r in peak_table[['proc_from', 'proc_to']].iterrows():
        start = r['proc_from']
        end = r['proc_to']

        # TODO need to use iterrows if not using index? just iterate over df?
        for _, row in truth.iterrows():
            if ((row['start'] >= start and row['start'] <= end) or
                (row['end'] >= start and row['end'] <= end)):

                true_sat_peaks.append(peak_id)
                break

    sat_peaks = set(sat_peaks)
    true_sat_peaks = set(true_sat_peaks)

    false_positives = sat_peaks - true_sat_peaks
    false_negatives = true_sat_peaks - sat_peaks
    true_positives = sat_peaks & true_sat_peaks

    if len(false_positives) > 0:
        print('False positives:', false_positives)

    if len(false_negatives) > 0:
        print('False negatives:', false_negatives)

    if len(true_positives) > 0:
        print('True positives:', true_positives)

    assert sat_peaks == true_sat_peaks

