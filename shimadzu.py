#!/usr/bin/env python3

"""
Function for parsing Shimadzu GCMSSolutions output files.

Generate these output files through the File->Export menu, or as part of a batch
job in the batch settings.
"""

import os
import sys
from io import StringIO
import traceback
import warnings

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
            if parts[0] == 'Raw':
                if '->' in parts:
                    return (float(parts[2]), float(parts[4]))
                else:
                    # TODO maybe rename from "range" to be inclusive of this
                    # case?
                    return float(parts[2])
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

    This table seems to need to be non-empty for any section for the
    similarity_search_results to appear at all.
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

    It seems that if the spectrum process table is empty, a section for this
    table will not appear, even if the box for this section is checked in the
    output options.
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


def available_sections():
    """Returns a dict of the section names here to the strings inside the square
    brackets starting each section in the Shimadzu ASCII output.
    """
    # TODO maybe just store ascii names, factor out a fn to normalize,
    # and normalize ascii names for (most of) these?
    sections = {
        'header': 'Header',
        'file_information': 'File Information',
        'sample_information': 'Sample Information',
        'original_files': 'Original Files',
        'file_description': 'File Description',
        'mc_peak_table': 'MC Peak Table',
        'ms_column_performance_table': 'MS Column Performance Table',
        'spectrum_process_table': 'Spectrum Process Table',
        'similarity_search_results': ('MS Similarity Search Results ' +
            'for Spectrum Process Table'),
        'ms_spectrum': 'MS Spectrum',
        'ms_chromatogram': 'MS Chromatogram'
    }
    # TODO check i'm not missing any that just use the stock parser...
    # TODO also print out their names where you check the boxes to get output
    # (which I think are different in some cases?) (in another fn)
    return sections


def read(filename, warn_missing_sections=False, skip_sections=[]):
    """
    skip_sections (list): A list of names of sections to skip. Either normalized
        or original names work.
    """
    if len(skip_sections) > 0:
        raise NotImplementedError

    # TODO TODO warn for each file that seems to be unannotated
    name2parser = {
        'mc_peak_table': mc_peak_table,
        'spectrum_process_table': spectrum_process_table,
        'similarity_search_results': similarity_search_results,
        'ms_column_performance_table': ms_column_performance_table,
        'ms_spectrum': MassSpectrum,
        'ms_chromatogram': Chromatogram
    }
    # TODO TODO fix case where header seems to be missing for some (first?)
    # sections
    required_tables = {
        'header',
        'sample_information',
        'mc_peak_table',
        'similarity_search_results'
    }
    with open(filename, 'r') as data:

        sample_data = dict()
        one_sample_data = dict()
        # TODO probably represent this another way...
        # TODO at least figure out relationsip between indices and ids (just off
        # by 1?)
        one_sample_data['ms_spectrum'] = []
        section_name = None
        sample_name = None
        skipping = False

        for line_num, line in enumerate(data):
            if line[0] == '[':
                # TODO derive sample name from header / sample info (+ maybe
                # check them against each other)
                if ((not skipping) and section_name is not None and not
                    (len(section_body) == 2 and section_body.strip() == '')):

                    parsed = None
                    try:
                        # TODO TODO support skipping certain sections w/
                        # None value (to parse remainder in case of error)
                        if section_name in name2parser:
                            parsed = name2parser[section_name](section_body)
                        else:
                            parsed = keyvalue_table(section_body)

                    except Exception as e:
                        quit = True
                        if isinstance(e, pd.errors.EmptyDataError):
                            if section_name in required_tables:
                                warnings.warn(('Skipping sample {} because it' +
                                    ' is missing required section {}.').format(
                                    sample_name, section_name))

                                one_sample_data = dict()
                                one_sample_data['ms_spectrum'] = []
                                section_name = None
                                sample_name = None
                                section_body = ''
                                skipping = True
                                # TODO maybe use a flag to still print the rest
                                # of the debug info before continuing
                                continue

                            elif warn_missing_sections:
                                # TODO TODO configure warning to fire more than
                                # once!
                                warnings.warn('Empty {}!'.format(section_name))
                            quit = False

                        if quit or warn_missing_sections:
                            print('')
                            if quit:
                                traceback.print_tb(e.__traceback__)
                            print(e, file=sys.stderr)
                            if quit:
                                print('')
                            print('Sample: {}'.format(sample_name))
                            print(('In section {}, ending at line {} in file ' +
                                '{}').format(section_name, line_num, filename))
                            if quit:
                                print('<section body>')
                                print(section_body)
                                print('<end of section body>')
                                sys.exit()

                    if parsed is not None:
                        if section_name == 'ms_spectrum':
                            one_sample_data['ms_spectrum'].append(parsed)

                        elif section_name in one_sample_data:
                            raise ValueError('duplicate "{}" section!'.format(
                                section_name))
                        else:
                            one_sample_data[section_name] = parsed
                            if section_name == 'sample_information':
                                # TODO TODO just convert all keyvalue tables to
                                # series to avoid having to index column first
                                # w/ number 1 
                                sample_name = one_sample_data[
                                    'sample_information'][1].at['sample_name']

                section_body = ''
                section_name = line[1:-2]

                if section_name == ('MS Similarity Search Results ' +
                    'for Spectrum Process Table'):

                    section_name = 'similarity_search_results'
                else:
                    section_name = section_name.lower().replace(' ', '_')

                if section_name == 'header':
                    skipping = False
                    if len(one_sample_data) > 1:
                        sample_data[sample_name] = one_sample_data
                        one_sample_data = dict()
                        one_sample_data['ms_spectrum'] = []
                        section_name = None
                        sample_name = None

            else:
                section_body += line

        # TODO TODO refactor so i don't have to repeat most of the logic of
        # the loop here to cover border case (last section, last sample)
        # TODO factor into like a try_parse function or something?
        if ((not skipping) and section_name is not None and not
            (len(section_body) == 2 and section_body.strip() == '')):

            parsed = None
            try:
                if section_name in name2parser:
                    parsed = name2parser[section_name](section_body)
                else:
                    parsed = keyvalue_table(section_body)

            except Exception as e:
                quit = True
                if isinstance(e, pd.errors.EmptyDataError):
                    if section_name in required_tables:
                        warnings.warn(('Skipping sample {} because it' +
                            ' is missing required section {}.').format(
                            sample_name, section_name))
                        # TODO maybe use a flag to still print the rest
                        # of the debug info before continuing
                        return

                    elif warn_missing_sections:
                        # TODO TODO configure warning to fire more than
                        # once!
                        warnings.warn('Empty {}!'.format(section_name))
                    quit = False

                if quit or warn_missing_sections:
                    print('')
                    if quit:
                        traceback.print_tb(e.__traceback__)
                    print(e, file=sys.stderr)
                    if quit:
                        print('')
                    print('Sample: {}'.format(sample_name))
                    print(('In section {}, ending at line {} in file ' +
                        '{}').format(section_name, line_num, filename))
                    if quit:
                        print('<section body>')
                        print(section_body)
                        print('<end of section body>')
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


def merge_spectrum_and_peak_tables(data_dict):
    """
    Takes a dict like the output of read and returns a pandas.DataFrame from
    merging the mc_peak_table, spectrum_process_table, and
    similarity_search_results.
    """
    raise NotImplementedError
    # TODO maybe also support merging column performance table or others
    # TODO rename (raw,top) and like to things like raw_top


def merge_simsearch_and_peak_tables(data_dict):
    """
    Takes a dict like the output of read and returns a pandas.DataFrame from
    merging the mc_peak_table and similarity_search_results. Uses the retention
    times in the spectrum process table to join spectra to peaks.
    """
    peak_df = data_dict['mc_peak_table']

    simsearch = data_dict['similarity_search_results']
    specprocess = data_dict['spectrum_process_table']

    ss_duplicate_spectra = simsearch[simsearch.duplicated()].index
    sp_duplicate_spectra = specprocess[specprocess.duplicated()].index

    # Sometimes there are duplicate rows in the sim search table that do not
    # correspond to duplicate rows in the spectrum process table.
    # These should not be dropped.
    ss_dup_set = set(ss_duplicate_spectra)
    for s in sp_duplicate_spectra:
        assert s in ss_dup_set

    ss_to_drop = \
        ss_duplicate_spectra[ss_duplicate_spectra.isin(sp_duplicate_spectra)]
    assert set(ss_to_drop) == set(sp_duplicate_spectra)

    if len(ss_to_drop) > 0:
        print('\ndropping from simsearch:')
        print(simsearch.loc[ss_to_drop])

    if len(sp_duplicate_spectra) > 0:
        print('dropping from specprocess:')
        print(specprocess.loc[sp_duplicate_spectra])
        print('')

    simsearch.drop(ss_to_drop, inplace=True)
    specprocess.drop(sp_duplicate_spectra, inplace=True)

    # TODO if there are cases where there are duplicates in one and not the
    # other, warn before dropping

    # TODO check there are no duplicates before this join
    # (if there are duplicates in end column, need some other key to match up
    # the spectra w/ the peaks...)
    simsearch_df = simsearch.join(specprocess['background'][['start','end']])

    sn = data_dict['sample_information'].at['sample_name',1]

    # Checking the rounding would not confuse peaks.
    assert not peak_df.duplicated(subset=['proc_from','proc_to']).any()
    assert not simsearch_df.duplicated(subset=['start','end']).any()

    df = pd.merge(peak_df.reset_index(), simsearch_df.reset_index(), how='left',
        left_on=['proc_from','proc_to'], right_on=['start','end'],
        validate='one_to_one')

    unmatched_spectra = (set(simsearch_df.index.unique()) - 
                         set(df.spectrum.unique()))

    # TODO could also try searching over peak w/o names in the same way,
    # in case some search results are not associated w/ integration peaks?
    unmatched_peaks = df[pd.notnull(df[['peak','name_x']]).all(axis=1) &
                         pd.isnull(df.spectrum)].index

    for s in unmatched_spectra:
        top = specprocess.loc[s, ('raw','top')]
        print('spectrum {}, peak @ {:.3f}'.format(s, top))
        print('spec name = {}'.format(simsearch_df.at[s, 'name']))

        matching_from = None
        matching_to = None
        matching_idx = None
        for i in unmatched_peaks:
            proc_from = df.at[i,'proc_from']
            proc_to = df.at[i,'proc_to']
            print(df.loc[i,['peak','proc_from','proc_to','name_x']])

            # TODO TODO handle case where multiple proc_<from/to> ranges
            # include this top (need to find smallest range)
            if proc_from <= top and top <= proc_to:
                if matching_idx is None:
                    print('first match found')
                    matching_from = proc_from
                    matching_to = proc_to
                    matching_idx = i

                # Existing match encloses the current integration range
                # (current integration range is more specific).
                elif matching_from <= proc_from and proc_to <= matching_to:
                    print('found smaller matching peak')
                    matching_from = proc_from
                    matching_to = proc_to
                    matching_idx = i

                # Current integration range encloses the existing match.
                elif proc_from <= matching_from and matching_to <= proc_to:
                    print('keeping existing match')
                    pass

                else:
                    assert False, ('Neither range enclosed other. Both ' +
                        'contained spectrum {} top'.format(s))

            else:
                print('peak not in range')

        if matching_idx is None:
            continue

        print('matching spectra {} to peak {}'.format(s,
            df.at[matching_idx, 'peak']))

        assert pd.isnull(df.at[matching_idx, 'spectrum']), \
            'this peak has already been assigned a spectrum'

        # Even if it doesn't correspond to a search result, each integration
        # range should have been assigned a spectrum in the spectrum process
        # table.
        # (but only looking at peaks with names, some spectra won't find
        # partners, so I can't assert this.)
        # Now, just checking after loop.
        #assert matching_idx is not None, \
        #    'no matching peak for spectrum {}'.format(s)

        # TODO more idiomatic way to merge row by row?
        for k, v in simsearch_df.loc[s].iteritems():
            if k + '_x' in df.columns:
                assert k not in df.columns
                k = k + '_y'

            if not (pd.isnull(df.at[matching_idx, k]) or
                    df.at[matching_idx, k] == v):
                # TODO why do some things seem to be false by default? suppress
                # output in (at least) that case?
                print('overwriting {}={} with {} for peak {}'.format(k,
                    df.at[matching_idx, k] == v, v,
                    df.at[matching_idx, 'peak']))

            df.at[matching_idx, k] = v

        df.at[matching_idx, 'spectrum'] = s

    # TODO TODO specifically test some cases where spectrum and peak #s don't
    # match up (i'm not overwriting spectrum column with peaks or something, am
    # i?)
    # Each peak has a spectrum, and vice versa.
    assert (pd.notnull(df.loc[pd.notnull(df.name_x), ['peak','spectrum']]
            ).all().all())
    # Each peak is only assigned to one spectrum, and vice versa.
    assert len(df) == len(df.peak.unique())
    assert (len(df[pd.notnull(df.name_x)]) ==
            len(df[pd.notnull(df.name_x)].spectrum.unique()))

    # TODO maybe just merge on name in case above fails?

    df['name_y'] = df.name_y.apply(lambda x: x.split('$$')[0].strip()
        if pd.notnull(x) else x)

    # TODO just use check name_y for assertion, but warn if x is not null and y
    # is null?
    name_match_rows = ((df.name_x == df.name_y) |
        pd.isnull(df[['name_x','name_y']]).all(axis=1))
    table_mismatch = not name_match_rows.all()

    verbose = True

    if table_mismatch:
        #####import ipdb; ipdb.set_trace()

        # TODO maybe print out all tables for mismatch and a few for
        # non-mismatch to see any patterns that might reveal cause?
        print('table mismatch at', sn)

        if verbose:
            print('\nMISMATCH BETWEEN NAMES IN PEAK / SPECTRUM TABLES!!')
            print(df.loc[~ name_match_rows, ['peak','spectrum','name_x',
                'name_y','proc_from','start','proc_to','end']])
            print('')

        assert False

    else:
        print('no table mismatch at', sn)

    if verbose and table_mismatch:
        print('Peak table:')
        print(peak_df[['ret_time','proc_from','proc_to','name',
            'mark']])
        print('Merged spectrum process and similarity search tables:')
        print(simsearch_df[['name','start','end']])
        print('\nSpectrum process table:')
        print(specprocess)
        # TODO maybe print out spectra not matched to peaks?
        # TODO keep and print rows w/ times from spectrum process table?
        print('Merged tables:')
        print(df[['peak','spectrum','proc_from','start','proc_to','end',
            'ret_time','name_x','name_y','mark']])

        # TODO print tables before dropping this stuff?
        print('simsearch to drop:', ss_to_drop)
        print('specprocess to drop:', sp_duplicate_spectra)
        print('')


    df.drop(columns=['name_y'], inplace=True)
    df.rename(columns={'name_x': 'name'}, inplace=True)

    # TODO assert if something was in simsearch_df, it was also in peak_df?
    # 1:1 validate above sufficient?
    df.drop(columns=['start','end'], inplace=True)

    return df, table_mismatch


def bound_saturation_threshold(cdf_filename, saturation_truth_csv,
    max_scan_jitter=6):
    """
    """
    import matplotlib.pyplot as plt

    from gcmstools.filetypes import AiaFile
    data = AiaFile(cdf_filename)
    scan_time = np.mean(np.diff(data.times))

    # All unlabeled masses in any labelled samples are assumed to be
    # unsaturated, so it is important to label all saturated peaks in a given
    # sample.
    # Columns: sample_name, start_(scan/rettime?), end_scan, m/z
    truth = pd.read_csv(saturation_truth_csv)

    bounds = []
    for start_jitter in range(-max_scan_jitter, max_scan_jitter + 1):
        for end_jitter in range(-max_scan_jitter, max_scan_jitter + 1):
            threshold_possible = True

            saturated = np.zeros_like(data.intensity, dtype=bool)
            for _, row in truth.iterrows():
                start = row['start']
                end = row['end']
                m_over_z = row['m/z']

                mz_index = np.argwhere(np.round(m_over_z) == data.masses)[0,0]

                # might want to also get scan before start?
                saturated[np.logical_and(data.times >= start + (start_jitter +
                    0.5) * scan_time, data.times <= end + (end_jitter + 0.5) *
                    scan_time), mz_index] = True

            tentative_bounds = []
            for mz in range(data.intensity.shape[1]):
                if saturated[:,mz].sum() > 0:
                    print(data.masses[mz])
                    upper_bound = np.min(data.intensity[saturated[:,mz], mz])
                    lower_bound = np.max(data.intensity[~ saturated[:,mz], mz])
                    tentative_bounds.append((data.masses[mz], lower_bound,
                        upper_bound))
                    '''
                    print(np.sort(data.intensity[saturated[:,mz],mz])[:10])
                    print(np.sort(data.intensity[~ saturated[:,mz],mz]
                        )[::-1][:10])
                    print(upper_bound)
                    print(lower_bound)
                    import ipdb
                    ipdb.set_trace()
                    '''
                    
                    '''
                    plt.plot(data.intensity[:, mz])
                    plt.plot(saturated[:,mz] * np.max(data.intensity[:, mz]) /
                        2.0)

                    plt.axhline(y=upper_bound, color='r',
                        label='Min sat. (upper bound)')
                    plt.axhline(y=lower_bound, color='g',
                        label='Max unsat. (lower bound)')

                    sat_indices = np.argwhere(saturated[:,mz])
                    unsat_indices = np.argwhere(~ saturated[:,mz])
                    au = sat_indices[
                        np.argmin(data.intensity[saturated[:,mz],mz]), 0]
                    al = unsat_indices[
                        np.argmax(data.intensity[~saturated[:,mz],mz]),0]

                    plt.axvline(x=au, color='r')
                    plt.axvline(x=al, color='g')

                    plt.legend()

                    plt.title('m/z = {}'.format(data.masses[mz]))
                    plt.show()
                    import ipdb
                    ipdb.set_trace()
                    '''

                    # TODO TODO TODO was this only failing by roughly an
                    # off-by-one on the time indexing (contract windows a little
                    # first?)
                    # TODO global threshold work in that case?
                    #assert upper_bound >= lower_bound, \
                    #    'one threshold would not work'

                    if upper_bound < lower_bound:
                        threshold_possible = False
                        break

            if not threshold_possible:
                print(('Threshold not possible with start jitter {}, and ' +
                    'end jitter {}').format(start_jitter, end_jitter))
                continue

            else:
                print(('Threshold POSSIBLE with start jitter {}, and ' +
                    'end jitter {}!!').format(start_jitter, end_jitter))
                bounds = tentative_bounds

    print(bounds)
    return bounds
    # TODO TODO TODO try m/z specific thresholds...

    #import ipdb
    #ipdb.set_trace()


def get_saturated_peaks(cdf_filename, peak_table):
    """
    Currently unclear whether it is possible to determine the 'saturation' the
    Shimadzu GUI indicates for a given mass spectrum from the intensity for each
    mass alone. The best case scenario, there would be a threshold X, wherever
    the intensity for a given mass equals or exceeds X, it is saturated in that
    scan.

    As the intensities for each mass do not seem to have a single maximal value,
    that threshold is either not the maximal value, or the particular maximal
    value also depends on other variables.
    """
    from gcmstools.filetypes import AiaFile
    data = AiaFile(cdf_filename)

    # TODO support taking just a list of pairs of start, stop retention time,
    # for more generality?
    sat_peaks = []

    # m/z dependent saturation? baseline? local sum? 
    threshold = 6500000

    for peak_id, r in peak_table[['proc_from', 'proc_to']].iterrows():
        start = r['proc_from']
        end = r['proc_to']

        # TODO how to efficiently get start and end?
        peak_indices = np.logical_and(data.times >= start, data.time <= end)
        peak_intensities = data.intensity[peak_indices, :]

        # TODO also print fraction of rows that are saturated?
        if np.any(peak_intensities >= threshold):
            sat_peaks.append(peak_id)

    return np.array(sat_peaks)


def print_peak_warnings(sample_data, min_similarity=93, peak_marks=True,
    cdf_directory=None):
    """
    For manual checking and correction.
    """
    # TODO TODO summary of files missing certain types of annotations
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
            if 'similarity_search_results' in data:
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
            else:
                warnings.warn('Can not print low similarity compounds, ' +
                    'because there was no similarity search results in ' +
                    'the output. Spectrum process table must be non-empty.')

        # TODO manually check labels between other two tables and
        # similarity_search_results seems right, since it doesn't have something
        # like the retention time to check automatically
        # TODO TODO allow spectrum process table to be less than length of
        # mc_peak_table, as it seems it doesn't always come into consideration
        # if you do some kind of manual integration (or maybe it was an
        # unannotated file???)
        if 'spectrum_process_table' in data:
            assert np.all(data['mc_peak_table'].index.unique() ==
                    data['spectrum_process_table'].index.unique())

            # data['mc_peak_table'].ret_time seems closest to "top" from
            # spectrum_process_table which, at least for the automatic peak
            # calling, is the same for both the background and raw columns in
            # that table.
            # TODO TODO change to using background, end ?
            assert np.allclose(data['mc_peak_table'].ret_time,
                data['spectrum_process_table']['raw','top'], atol=5e-3)
            # TODO (above) are these the appropriate columns to match up, if
            # this atol is required?

        unidentified = \
            data['mc_peak_table'][data['mc_peak_table'].name.isnull()].index

        if len(unidentified) > 0:
            print('Peaks that do not have compounds assigned:')
            for i in unidentified:
                print(i)


        if cdf_directory is not None:
            # TODO break this stuff into funcion wrapping CDF loading to just
            # add that data to each entry in sample_data dict?
            qgd_filename = \
                data['header'].at['data_file_name',1].split('\\')[-1]
            cdf_export = os.path.join(cdf_directory, qgd_filename[:-4] + '.CDF')

            sat_peaks = get_saturated_peaks(cdf_export, data['mc_peak_table'])

            print('Likely saturated peaks:')
            for i in sat_peaks:
                print(i)


        print('\n' + ('#' * lwidth))


def standardize(sample_data, standard_data, thru_origin=True):
    """
    Args:
    sample_data (dict): returned by read function, with the input file
        containing all data on the unknown mixtures.
    standard_data (dict): returned by read function, with the input file
        containing all data from external standards.
    thru_origin (bool): (default=True) If True, forces calibration curve through
        (0, 0).

    Returns a dict like sample_data, but with a "concentration" column added to
    each table with an "area" column.
    """
    # TODO support picking out samples that are supposed to be standards with
    # flags built in to shimadzu software?
    # TODO support loading calibration output from their software into the same
    # format this returns
    raise NotImplementedError

    chem2curve = dict()
    for name, data in standard_data.items():
        # TODO sufficient to use CAS in similarity search table to normalize
        # within CAS returned from this similarity search?
        # or is it not-invertible, as (name->pubchem search->CAS) seemed to be?
        #chem_id = chemutils.name2inchi(name)
        #chem2curve[chem_id] = 
        import ipdb
        ipdb.set_trace()
        assert len(data['mc_peak_table']) == 1
        #assert len(data['ms


    standardized_data = dict()
    

    # TODO TODO standardize within tuning file. within anything else?

    return standardized_data


# TODO make enclosing class w/ all of the data from a run? put it all in a big
# table across runs (well, most?)?

# TODO change shimadzu "fragment table" settings s.t. chromatogram doesn't
# export w/ redundant "relative area" columns. do w/ other tables to extent
# possible.

# TODO could check all sample file names are equal to sample names


if __name__ == '__main__':
    sample_data = read('example_data/181026_mango_N3_split2_onlyTICMIC.txt')

    print_peak_warnings(sample_data, cdf_directory='example_data')

    import ipdb
    ipdb.set_trace()

    standard_data = read('example_data/run2_dan_standards.txt')
    sample_data = standardize(sample_data, standard_data)

    # TODO TODO try to calibrate samples with one m/z peak saturated, from other
    # peaks
    # TODO TODO TODO find a good test case in existing data (fit with all/minor
    # peaks -> test agreement of concentration determination)
    # TODO TODO come up w/ test to also make sure this works in saturated case

