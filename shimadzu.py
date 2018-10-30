#!/usr/bin/env python3

"""
Function for parsing Shimadzu GCMSSolutions output files.

Generate these output files through the File->Export menu, or as part of a batch
job in the batch settings.
"""

import sys
from io import StringIO

import numpy as np
import pandas as pd


'''
class Sample:
    def __init__(self, 
'''


class MassSpectrum:
    # TODO implement __str__ method
    def __init__(self, body):
        """
        body (str): the section of the ASCII output that corresponds to this
        table, without the first line with the section name in square brackets
        """
        # TODO more efficient way to index first three lines w/o processing full
        # table? (maybe always work w/ lists and just concat relevant lines
        # before df parsing?)
        lines = body.split('\n')

        # TODO is "spectrum range" an appropriate way to describe the
        # information in these lines?
        def parse_spectrum_range(s):
            parts = s.split()
            # TODO what are units for other scan ranges besides "scan"?
            if parts[0] == 'Raw':
                return (float(parts[2]), float(parts[4]))
            else:
                return (float(parts[1]), float(parts[3]))

        # TODO what is meaning of raw / "background" spectrum? how are they
        # used?
        self.raw_spectrum_range = parse_spectrum_range(lines[1])
        self.background_spectrum_range = parse_spectrum_range(lines[2])

        # TODO why does "Event# <N>" (line before MS spectrum table header)
        # always have N=1? just toss this line?

        self.df = pd.read_table(StringIO(body), skiprows=5).rename(
            columns=normalize_name).set_index('m/z').drop(
            columns='relative_intensity')


# TODO TODO maybe pack all MassSpectrum tables into a dataframe with a
# multiindex on the rows, where the first index is the peak #?
# otherwise, how do i want to associate peak # from chromatogram with spectra?

class Chromatogram:
    # TODO implement __str__ method
    def __init__(self, body):
        """
        body (str): the section of the ASCII output that corresponds to this
        table, without the first line with the section name in square brackets
        """
        lines = body.split('\n')
        # TODO want other lines?
        self.event_time_ms = int(lines[1].split()[-1])

        self.df = pd.read_table(StringIO(body), skiprows=5,
            usecols=[0,1]).rename(columns=normalize_name).set_index('ret_time')


def mc_peak_table(body):
    """
    body (str): the section of the ASCII output that corresponds to this table,
    without the first line with the section name in square brackets
    """
    # TODO want to record "Mass TIC" line, for applicability in cases where TIC
    # is some other value?
    df = pd.read_table(StringIO(body), skiprows=2).rename(
        columns=normalize_name).set_index('peak')
    df['mark'] = df['mark'].apply(lambda m: m.strip())
    # "Mark" column character code meanings:
    #  S: The main peak used to process tailing.
    #  T: Peaks which have undergone tailing processing.
    #  L: The first (leading) tailing processing peak.
    #  V: For unresolved peaks, the second peak and greater.
    #  E: An error peak.
    #  MI: Manually integrated peak
    return df


def ms_column_performance_table(body):
    """
    body (str): the section of the ASCII output that corresponds to this table,
    without the first line with the section name in square brackets
    """
    # TODO TODO so can this be used to understand the tailing processing?
    # for manually annotated peaks too?
    # TODO replace '---' w/ nan? need to replace other values w/ nan elsewhere?
    df = pd.read_table(StringIO(body), skiprows=1).rename(
        columns=normalize_name).set_index('peak')
    return df


def spectrum_process_table(body):
    """
    body (str): the section of the ASCII output that corresponds to this table,
    without the first line with the section name in square brackets
    """
    df = pd.read_table(StringIO(body), skiprows=2).set_index('Spectrum#')
    # TODO make sure to be consistent each time same id is used
    df.index.name = 'spectrum'
    # TODO come up with more appropriate 'names' here
    df.columns = pd.MultiIndex.from_product([('raw','background'),
        ('top','start','end')], names=('type','point'))
    return df


# TODO just merge this w/ spectrum process table?
def similarity_search_results(body):
    """
    body (str): the section of the ASCII output that corresponds to this table,
    without the first line with the section name in square brackets
    """
    return pd.read_table(StringIO(body), skiprows=1).rename(
        columns=normalize_name).set_index('spectrum')


def keyvalue_table(body):
    """Default parsing functions, for all the basic metadata sections to the
    Shimadzu output.

    body (str): the section of the ASCII output that corresponds to this table,
    without the first line with the section name in square brackets
    """
    return pd.read_table(StringIO(body), header=None
        ).set_index(0).rename(index=normalize_name)


def normalize_name(n):
    # To avoid double underscores in this case.
    n = n.replace('. ', '.')
    if n[-1] == '.':
        n = n[:-1]
    n = n.replace(' ','_').replace('.','_').replace('#','').lower()
    # TODO also need to strip leading trailing whitespace after replacing?
    if n[-1] == '_':
        n = n[:-1]
    return n


def read(filename):
    """
    """
    name2parser = {
        'mc_peak_table': mc_peak_table,
        'spectrum_process_table': spectrum_process_table,
        'similarity_search_results': similarity_search_results,
        'ms_column_performance_table': ms_column_performance_table,
        'ms_spectrum': MassSpectrum,
        'ms_chromatogram': Chromatogram
    }
    with open(filename, 'r') as data:

        sample_data = dict()
        one_sample_data = dict()
        # TODO probably represent this another way...
        # TODO at least figure out relationsip between indices and ids (just off
        # by 1?)
        one_sample_data['ms_spectrum'] = []
        section_name = None

        for line_num, line in enumerate(data):
            # TODO need to modify this s.t. we can actually parse last section
            # (|| EOF?)
            if line[0] == '[':
                # TODO TODO TODO derive sample name from header / sample info (+
                # maybe check them against each other)
                if (section_name is not None and not
                    (len(section_body) == 2 and section_body.strip() == '')):

                    parsed = None
                    if section_name in name2parser:
                        parsed = name2parser[section_name](section_body)

                    else:
                        try:
                            parsed = keyvalue_table(section_body)
                        except Exception as e:
                            print(e)
                            # TODO also get section start line?
                            print('At line {}'.format(line_num))
                            print(section_name)
                            print(section_body)
                            sys.exit()

                    if parsed is not None:
                        if section_name == 'ms_spectrum':
                            one_sample_data['ms_spectrum'].append(parsed)

                        elif section_name in one_sample_data:
                            raise ValueError('duplicate "{}" section!'.format(
                                section_name))
                        else:
                            one_sample_data[section_name] = parsed

                section_body = ''
                section_name = line[1:-2]
                if len(one_sample_data) > 1 and section_name == 'Header':
                    # TODO TODO just convert all keyvalue tables to series to
                    # avoid having to index column first w/ number 1 
                    sample_name = one_sample_data['sample_information'][1
                        ].at['sample_name']

                    sample_data[sample_name] = one_sample_data
                    one_sample_data = dict()
                    one_sample_data['ms_spectrum'] = []

                if section_name == ('MS Similarity Search Results ' +
                    'for Spectrum Process Table'):

                    section_name = 'similarity_search_results'
                else:
                    section_name = section_name.lower().replace(' ', '_')

            else:
                section_body += line

        # TODO TODO refactor so i don't have to repeat most of the logic of
        # the loop here to cover border case (last section, last sample)
        # TODO factor into like a try_parse function or something?
        if (section_name is not None and not
            (len(section_body) == 2 and section_body.strip() == '')):

            parsed = None
            if section_name in name2parser:
                parsed = name2parser[section_name](section_body)

            else:
                try:
                    parsed = keyvalue_table(section_body)
                except Exception as e:
                    print(section_body)
                    print('In table {}, ending at line {}'.format(section_name,
                        line_num))
                    print(e)
                    sys.exit()

            if parsed is not None:
                if section_name == 'ms_spectrum':
                    one_sample_data['ms_spectrum'].append(parsed)

                elif section_name in one_sample_data:
                    raise ValueError('duplicate "{}" section!'.format(
                        section_name))
                else:
                    one_sample_data[section_name] = parsed

        if len(one_sample_data) > 1:
            sample_name = one_sample_data['sample_information'][1
                ].at['sample_name']
            sample_data[sample_name] = one_sample_data

    return sample_data


def print_peak_warnings(sample_data, min_similarity=93, peak_marks=True,
    saturation=True):
    """
    For manual checking and correction.
    """
    lwidth = 80
    def print_marked(name, mark):
        marked = marks[marks.apply(lambda x: mark in x)]
        if len(marked) > 0:
            print('{} peaks:'.format(name.title()))
            # TODO TODO print ret. times, if there is no way to see peak id in
            # gui (go to table -> click on it?)
            for i in marked.index:
                print(i)
            print('')

    print('#' * lwidth)
    for sample_name in sorted(list(sample_data.keys())):
        print(sample_name)
        data = sample_data[sample_name]

        if peak_marks:
            marks = data['mc_peak_table']['mark']
            assert marks.apply(lambda x: 'L' in x).sum() == 0, \
                'Not sure how to handle "L" peak mark'

            print_marked('"unresolved"', 'V')
            print_marked('primary (tail processing)', 'S')
            print_marked('secondary (tail processing)', 'T')
            print_marked('"error"', 'E')
            print_marked('manual', 'MI')

        # TODO plot spectra for each peak w/ extra processing to check it is
        # reasonable?

        # TODO TODO somehow figure out whether each peak is saturated somewhere
        # within its bounds or not
        # TODO will i be able to call saturation w/o raw mass spectra output?
        # (both TIC and some methods for getting spectra will underestimate
        # max...)
        # TODO could output MC for each m/z? or just read data file? convert to
        # CDF and read that way?

        if min_similarity > 0:
            simsearch = data['similarity_search_results']
            low_similarity = simsearch[(simsearch['hit'] == 1) &
                (simsearch['si'] < min_similarity)]

            # TODO sort from lowest sim to highest?
            if len(low_similarity) > 0:
                print('Peaks with SI less than {}:'.format(min_similarity))
                for i, r in low_similarity.iterrows():
                    # TODO fixed width
                    print('{} - {}'.format(i, r['si']))
                print('')

        # TODO manually check labels between other two tables and
        # similarity_search_results seems right, since it doesn't have something
        # like the retention time to check automatically
        assert np.all(data['mc_peak_table'].index.unique() ==
                data['spectrum_process_table'].index.unique())

        # data['mc_peak_table'].ret_time seems closest to "top" from
        # spectrum_process_table which, at least for the automatic peak calling,
        # is the same for both the background and raw columns in that table.
        assert np.allclose(data['mc_peak_table'].ret_time,
            data['spectrum_process_table']['raw','top'], atol=5e-3)
        # TODO (above) are these the appropriate columns to match up, if this
        # atol is required?

        unidentified = \
            data['mc_peak_table'][data['mc_peak_table'].name.isnull()].index

        if len(unidentified) > 0:
            print('Peaks that do not have compounds assigned:')
            for i in unidentified:
                print(i)

        print('\n' + ('#' * lwidth))


# TODO make enclosing class w/ all of the data from a run? put it all in a big
# table across runs (well, most?)?

# TODO change shimadzu "fragment table" settings s.t. chromatogram doesn't
# export w/ redundant "relative area" columns. do w/ other tables to extent
# possible.

# TODO could check all sample file names are equal to sample names

# TODO TODO standardize within tuning file. within anything else?

if __name__ == '__main__':
    data = read('example_data/ASCIIData_min_sim_92.txt')

    print_peak_warnings(data)

    # TODO TODO try to calibrate samples with one m/z peak saturated, from other
    # peaks
    # TODO TODO TODO find a good test case in existing data (fit with all/minor
    # peaks -> test agreement of concentration determination)
    # TODO TODO come up w/ test to also make sure this works in saturated case

