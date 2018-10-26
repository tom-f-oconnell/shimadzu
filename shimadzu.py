#!/usr/bin/env python3

"""
Function for parsing Shimadzu GCMSSolutions output files.

Generate these output files through the File->Export menu, or as part of a batch
job in the batch settings.
"""

import sys
from io import StringIO

import pandas as pd


'''
class Sample:
    def __init__(self, 
'''


class MassSpectrum:
    def __init__(self, section_body):
        # TODO more efficient way to index first three lines w/o processing full
        # table? (maybe always work w/ lists and just concat relevant lines
        # before df parsing?)
        lines = section_body.split('\n')

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

        self.df = pd.read_table(StringIO(section_body), skiprows=5).rename(
            columns=normalize_name).set_index('m/z').drop(
            columns='relative_intensity')


# TODO TODO maybe pack all MassSpectrum tables into a dataframe with a
# multiindex on the rows, where the first index is the peak #?
# otherwise, how do i want to associate peak # from chromatogram with spectra?

class Chromatogram:
    def __init__(self, body):
        lines = body.split('\n')
        # TODO want other lines?
        self.event_time_ms = int(lines[1].split()[-1])

        self.df = pd.read_table(StringIO(body), skiprows=5,
            usecols=[0,1]).rename(columns=normalize_name).set_index('ret_time')


def mc_peak_table(body):
    """
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


def spectrum_process_table(body):
    """
    """
    df = pd.read_table(StringIO(body), skiprows=2).set_index('Spectrum#')
    # TODO make sure to be consistent each time same id is used
    df.index.name = 'spectrum'
    # TODO come up with more appropriate 'names' here
    df.columns = pd.MultiIndex.from_product([('raw','background'),
        ('top','start','end')], names=('type','point'))
    return df


# TODO TODO just merge this w/ spectrum process table?
def similarity_search_results(body):
    """
    """
    return pd.read_table(StringIO(body), skiprows=1).rename(
        columns=normalize_name).set_index('spectrum')


def keyvalue_table(body):
    """Default parsing functions, for all the basic metadata sections to the
    Shimadzu output.
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
        'ms_spectrum': MassSpectrum,
        'ms_chromatogram': Chromatogram,
    }
    with open(filename, 'r') as data:

        all_data = dict()
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

                    all_data[sample_name] = one_sample_data
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
                    print(e)
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

        if len(one_sample_data) > 1:
            sample_name = one_sample_data['sample_information'][1
                ].at['sample_name']
            all_data[sample_name] = one_sample_data

    return all_data


def print_peak_warnings(all_data, min_similarity=0, peak_marks=True,
    saturation=True):
    """
    For manual checking and correction.
    """
    lwidth = 80
    def print_marked(name, mark):
        marked = marks[marks.apply(lambda x: mark in x)]
        if len(marked) > 0:
            print('{} {} peaks:'.format(len(marked), name))
            # TODO TODO print ret. times, if there is no way to see peak id in
            # gui (go to table -> click on it?)
            for i in marked.index:
                print(i)
            print('')

    print('#' * lwidth)
    for sample_name, data in all_data.items():
        print(sample_name)

        marks = data['mc_peak_table']['mark']
        assert marks.apply(lambda x: 'L' in x).sum() == 0, \
            'Not sure how to handle "L" peak mark'

        print('{} peaks'.format(len(marks)))
        print_marked('"unresolved"', 'V')
        print_marked('primary (tail processing)', 'S')
        print_marked('secondary (tail processing)', 'T')
        print_marked('"error"', 'E')
        print_marked('manual', 'MI')
        print('#' * lwidth)

    # TODO plot spectra for each peak w/ extra processing to check it is
    # reasonable?

    # TODO TODO somehow figure out whether each peak is saturated somewhere
    # within its bounds or not

    # TODO kwarg for similarity cutoff to report, 
    # TODO what actually happens if no match is found, if that's not
    # "unresolved"?



# TODO make enclosing class w/ all of the data from a run? put it all in a big
# table across runs (well, most?)?

# TODO change shimadzu "fragment table" settings s.t. chromatogram doesn't
# export w/ redundant "relative area" columns. do w/ other tables to extent
# possible.

# TODO could check all sample file names are equal to sample names

if __name__ == '__main__':
    data = read('data/example/ASCIIData.txt')

    print_peak_warnings(data)

    # TODO TODO try to calibrate samples with one m/z peak saturated, from other
    # peaks
    # TODO TODO TODO find a good test case in existing data (fit with all/minor
    # peaks -> test agreement of concentration determination)
    # TODO TODO come up w/ test to also make sure this works in saturated case

