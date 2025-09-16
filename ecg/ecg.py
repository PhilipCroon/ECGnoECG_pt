# -*- coding: utf-8 -*-
"""ECG (waveform) Dicom module

Read and plot images from DICOM ECG waveforms.
"""

"""
The MIT License (MIT)

Copyright (c) 2013 Marco De Benedetto <debe@galliera.it>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""
import numpy as np
import pydicom as dicom

import struct
import io
from io import BytesIO
import os
import requests
from . import i18n
import re
from datetime import datetime
from matplotlib import use
import wfdb
from scipy.signal import butter, lfilter, resample
import gc
from scipy.ndimage import median_filter
import xml.etree.ElementTree as ET
from math import ceil 
from matplotlib.ticker import AutoMinorLocator
import os
from math import ceil
import gc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from ecgconfig import WADOSERVER, LAYOUT, INSTITUTION
except ImportError:
    WADOSERVER = "http://example.com"
    LAYOUT = {'3x4_1': [[0, 3, 6, 9],
                        [1, 4, 7, 10],
                        [2, 5, 8, 11],
                        [1]],
              '3x4': [[0, 3, 6, 9],
                      [1, 4, 7, 10],
                      [2, 5, 8, 11]],
              '6x2': [[0, 6],
                      [1, 7],
                      [2, 8],
                      [3, 9],
                      [4, 10],
                      [5, 11]],
              '12x1': [[0],
                       [1],
                       [2],
                       [3],
                       [4],
                       [5],
                       [6],
                       [7],
                       [8],
                       [9],
                       [10],
                       [11]]}

    # If INSTITUTION is set to None the value of the tag InstitutionName is
    # used

    INSTITUTION = None

__author__ = "Marco De Benedetto and Simone Ferretti"
__license__ = "MIT"
__credits__ = ["Marco De Benedetto", "Simone Ferretti", "Francesco Formisano"]
__email__ = "debe@galliera.it"


def butter_lowpass(highcut, sampfreq, order):
    """Supporting function.

    Prepare some data and actually call the scipy butter function.
    """

    nyquist_freq = .5 * sampfreq
    high = highcut / nyquist_freq
    num, denom = butter(order, high, btype='lowpass')
    return num, denom


def butter_lowpass_filter(data, highcut, sampfreq, order):
    """Apply the Butterworth lowpass filter to the DICOM waveform.

    @param data: the waveform data.
    @param highcut: the frequencies from which apply the cut.
    @param sampfreq: the sampling frequency.
    @param order: the filter order.
    """

    num, denom = butter_lowpass(highcut, sampfreq, order=order)
    return lfilter(num, denom, data)


class ECG(object):
    """The class representing the ECG object
    """

    paper_w, paper_h = 297.0, 210.0

    # Dimensions in mm of plot area
    width = 250.0
    height = 170.0
    margin_left = margin_right = .5 * (paper_w - width)
    margin_bottom = 10.0

    # Normalized in [0, 1]
    left = margin_left / paper_w
    right = left + width / paper_w
    bottom = margin_bottom / paper_h
    top = bottom + height / paper_h

    def __init__(self, source, info_print=True):
        """The ECG class constructor.

        @param source: the ECG source, it could be a filename, a buffer
        or a dict of study, serie, object info (to query
        a WADO server).
        @type source: C{str} or C{dict}.
        """
        self.info_print = info_print
        def err(msg):
            raise Exception

        def wadoget(stu, ser, obj):
            """Query the WADO server.

            @return: a buffer containing the DICOM object (the WADO response).
            @rtype: C{cStringIO.StringIO}.
            """
            payload = {
                'requestType': 'WADO',
                'contentType': 'application/dicom',
                'studyUID': stu,
                'seriesUID': ser,
                'objectUID': obj
            }
            headers = {'content-type': 'application/json'}

            resp = requests.get(WADOSERVER, params=payload, headers=headers)
            return io.BytesIO(resp.content)

        if isinstance(source, dict):
            # dictionary of stu, ser, obj
            if set(source.keys()) == set(('stu', 'ser', 'obj')):
                inputdata = wadoget(**source)
            else:
                err("source must be a dictionary of stu, ser and obj")
        elif isinstance(source, str) or getattr(source, 'getvalue'):
            # it is a filename or a (StringIO or cStringIO buffer)
            inputdata = source                  
        else:
            # What is it?
            err("'source' must be a path/to/file.ext string\n" +
                "or a dictionary of stu, ser and obj")
            
        if isinstance(source, str) and \
                        (source.endswith('.hea') or source.endswith('.dat')):
                self.read_physio(inputdata)
            
        else:
            self.read_dicom(inputdata)
        
        self.fig, self.axis = self.create_figure()


    def read_dicom(self, inputdata):
        """If the file is a dicom, extract all the information"""
        source = BytesIO(open(inputdata, mode='rb').read())
        try:
            self.dicom = dicom.dcmread(source)
            print('ok')
            """@ivar: the dicom object."""
        except dicom.filereader.InvalidDicomError as err:
            raise ECGReadFileError(err)

        sequence_item = self.dicom.WaveformSequence[0]

        assert (sequence_item.WaveformSampleInterpretation == 'SS')
        assert (sequence_item.WaveformBitsAllocated == 16)

        self.channel_definitions = sequence_item.ChannelDefinitionSequence
        self.wavewform_data = sequence_item.WaveformData
        self.channels_no = sequence_item.NumberOfWaveformChannels
        self.samples = sequence_item.NumberOfWaveformSamples
        self.sampling_frequency = sequence_item.SamplingFrequency
        self.duration = self.samples / self.sampling_frequency
        if self.duration < 10.0:
            raise ValueError(
                f"ECG waveform is too short ({self.duration:.2f} s). Must be ≥ 10 s."
            )
        # ────────
        self.mm_s = self.width / self.duration
        self.signals = self._signals()
        self._check_flat_tail(self.signals)
        self.resample_to_500hz()

    def read_physio(self, inputdata):
        """If the file is a .hea/.dat, extract all the information."""
        try:
            # Load the record; assumes the path without the '.hea' or '.dat' extension
            fname, _ = os.path.splitext(inputdata)
            self.record = wfdb.rdrecord(fname)
        except ValueError as err:
            # Handle errors in reading .hea/.dat files
            raise ECGReadFileError(err)

        # Check that the file contains the expected number of leads, e.g., 12 for a standard ECG
        assert self.record.n_sig == 12, "Expected 12 leads in the ECG."
        self.info_print = False
        self.waveform_data = self.record.p_signal
        self.channel_names = self.record.sig_name
        self.channel_definitions = False
        self.channels_no = self.record.n_sig
        self.samples = self.record.sig_len
        self.sampling_frequency = self.record.fs
        self.duration = self.samples / self.sampling_frequency
        self.mm_s = self.width / self.duration
        self.signals = self._signalsphysio()

        # Additional assertions or checks could be added here, for example:
        assert self.record.fs > 0, "Sampling frequency must be greater than zero."

    def __del__(self):
        """
        Figures created through the pyplot interface
        (`matplotlib.pyplot.figure`) are retained until explicitly
        closed and may consume too much memory.
        """

        plt.cla()
        plt.clf()
        plt.close()

    def create_figure(self, border: bool = False):
        """
        Prepare a new figure and axes, optionally removing all white margins.

        Parameters
        ----------
        borderless : bool
            If True, collapse all margins so that the axes fill the entire figure.
            If False, use the default self.left/right/top/bottom settings.
        """
        fig, axes = plt.subplots()

        if border == False:
            # push the axes to the very edges of the canvas
            fig.subplots_adjust(left=0.0, right=1.0, top=1.0, bottom=0.0)
        else:
            # keep the original margins defined by self.left/right/top/bottom
            fig.subplots_adjust(left=self.left,
                                right=self.right,
                                top=self.top,
                                bottom=self.bottom)

        axes.set_ylim([0, self.height])
        axes.set_xlim([0, self.samples - 1])
        return fig, axes

    def _signals(self):
        """
        Retrieve the signals from the DICOM WaveformData object.

        sequence_item := dicom.dataset.FileDataset.WaveformData[n]

        @return: a list of signals.
        @rtype: C{list}
        """

        factor = np.zeros(self.channels_no) + 1
        baseln = np.zeros(self.channels_no)
        units = []
        for idx in range(self.channels_no):
            definition = self.channel_definitions[idx]

            assert (definition.WaveformBitsStored == 16)

            if definition.get('ChannelSensitivity'):
                factor[idx] = (
                    float(definition.ChannelSensitivity) *
                    float(definition.ChannelSensitivityCorrectionFactor)
                )

            if definition.get('ChannelBaseline'):
                baseln[idx] = float(definition.get('ChannelBaseline'))

            units.append(
                definition.ChannelSensitivityUnitsSequence[0].CodeValue
            )

        unpack_fmt = '<%dh' % (len(self.wavewform_data) / 2)
        unpacked_waveform_data = struct.unpack(unpack_fmt, self.wavewform_data)
        signals = np.asarray(
            unpacked_waveform_data,
            dtype=np.float32).reshape(
            self.samples,
            self.channels_no).transpose()

        for channel in range(self.channels_no):
            signals[channel] = (
                (signals[channel] + baseln[channel]) * factor[channel]
            )

        high = 40.0

        # conversion factor to obtain millivolts values
        millivolts = {'uV': 1000.0, 'mV': 1.0}

        for i, signal in enumerate(signals):
            signals[i] = butter_lowpass_filter(
                np.asarray(signal),
                high,
                self.sampling_frequency,
                order=2
            ) / millivolts[units[i]]

        return signals
    
    
    def _signalsphysio(self):
        """
        Retrieve the signals from the Physionet object.

        @return: a list of signals.
        @rtype: C{list}
        """

        signals = np.asarray(
            self.waveform_data,
            dtype=np.float32).reshape(
            self.samples,
            self.channels_no).transpose()

        high = 40.0

        for i, signal in enumerate(signals):
            signals[i] = butter_lowpass_filter(
                np.asarray(signal),
                high,
                self.sampling_frequency,
                order=2
            )

        return signals
    
    def _check_flat_tail(self, signal, flat_len_sec=1.0, tol=1e-6):
        """
        Check for trailing flat (zero) padding in ECG signal and fix sampling frequency if needed.

        Parameters
        ----------
        signal : np.ndarray
            ECG signal array of shape (channels, samples)
        flat_len_sec : float
            Minimum duration of flat signal (in seconds) to consider it padding.
        tol : float
            Tolerance for considering values as zero.
        
        Actions:
        - If the last `flat_len_sec` seconds are flat and signal has 5000 samples,
        trims last 1000 samples and updates sampling_frequency to 400.
        """
        expected_padding_samples = int(500 * flat_len_sec)
        total_samples = signal.shape[1]

        # Check if signal is 12-lead: (12, samples)
        if total_samples == 5000:
            tail = signal[:, -expected_padding_samples:]
            if np.all(np.abs(tail) < tol):
                print("Flat tail detected: trimming last 1000 samples and correcting sampling frequency.")
                self.signals = signal[:, :4000]
                self.samples = 4000
                self.duration = self.samples / 400.0
                self.sampling_frequency = 400
                self.mm_s = self.width / self.duration
            else:
                print("No flat tail detected. Keeping original signal.")
        else:
            print("Signal length not equal to 5000. No trimming applied.")
            
        from scipy.signal import resample


    def resample_to_500hz(self) -> np.ndarray:
        """
        Resample self.signals (shape: 12 x N) to 500 Hz and 5000 samples.

        Returns:
            np.ndarray: Resampled ECG signal with shape (12, 5000).
        """
        target_fs = 500.0
        target_samples = int(self.signals.shape[1] * (target_fs / self.sampling_frequency))
        self.signals = resample(self.signals, target_samples, axis=1)

        # Enforce fixed length of 5000 samples
        if self.signals.shape[1] != 5000:
            self.signals = resample(self.signals, 5000, axis=1)

        # Update sampling frequency
        self.sampling_frequency = target_fs

        return self.signals

    def draw_grid(self, minor_axis):
        """Draw the grid in the ecg plotting area."""

        if minor_axis:
            self.axis.xaxis.set_minor_locator(
                plt.LinearLocator(int(self.width + 1))
            )
            self.axis.yaxis.set_minor_locator(
                plt.LinearLocator(int(self.height + 1))
            )

        self.axis.xaxis.set_major_locator(
            plt.LinearLocator(int(self.width / 5 + 1))
        )
        self.axis.yaxis.set_major_locator(
            plt.LinearLocator(int(self.height / 5 + 1))
        )

        color = {'minor': '#ff5333', 'major': '#d43d1a'}
        linewidth = {'minor': .1, 'major': .2}

        for axe in 'x', 'y':
            for which in 'major', 'minor':
                self.axis.grid(
                    which=which,
                    axis=axe,
                    linestyle='-',
                    linewidth=linewidth[which],
                    color=color[which]
                )

                self.axis.tick_params(
                    which=which,
                    axis=axe,
                    color=color[which],
                    bottom=False,
                    top=False,
                    left=False,
                    right=False
                )

        self.axis.set_xticklabels([])
        self.axis.set_yticklabels([])

    def legend(self):
        """A string containing the legend.

        Auxiliary function for the print_info method.
        """

        if not hasattr(self.dicom, 'WaveformAnnotationSequence'):
            return ''

        ecgdata = {}
        for was in self.dicom.WaveformAnnotationSequence:
            if was.get('ConceptNameCodeSequence'):
                cncs = was.ConceptNameCodeSequence[0]
                if cncs.CodeMeaning in (
                        'QT Interval',
                        'QTc Interval',
                        'RR Interval',
                        'VRate',
                        'QRS Duration',
                        'QRS Axis',
                        'T Axis',
                        'P Axis',
                        'PR Interval'
                ):
                    ecgdata[cncs.CodeMeaning] = str(was.NumericValue)

        # If VRate is not defined we calculate ventricular rate from
        # RR interval
        try:
            vrate = float(ecgdata.get('VRate'))
        except (TypeError, ValueError):
            try:
                vrate = "%.1f" % (
                    60.0 / self.duration *
                    self.samples / float(ecgdata.get('RR Interval'))
                )
            except (TypeError, ValueError, ZeroDivisionError):
                vrate = "(unknown)"

        ret_str = "%s: %s BPM\n" % (i18n.ventr_freq, vrate)
        ret_str_tmpl = "%s: %s ms\n%s: %s ms\n%s: %s/%s ms\n%s: %s %s %s"

        ret_str += ret_str_tmpl % (
            i18n.pr_interval,
            ecgdata.get('PR Interval', ''),
            i18n.qrs_duration,
            ecgdata.get('QRS Duration', ''),
            i18n.qt_qtc,
            ecgdata.get('QT Interval', ''),
            ecgdata.get('QTc Interval', ''),
            i18n.prt_axis,
            ecgdata.get('P Axis', ''),
            ecgdata.get('QRS Axis', ''),
            ecgdata.get('T Axis', '')
        )

        return ret_str

    def interpretation(self):
        """Return the string representing the automatic interpretation
        of the study.
        """

        if not hasattr(self.dicom, 'WaveformAnnotationSequence'):
            return ''

        ret_str = ''
        for note in self.dicom.WaveformAnnotationSequence:
            if hasattr(note, 'UnformattedTextValue'):
                if note.UnformattedTextValue:
                    ret_str = "%s\n%s" % (
                        ret_str,
                        note.UnformattedTextValue
                    )

        return ret_str

    def print_info(self, interpretation):
        """Print info about the patient and about the ecg signals.
        """

        try:
            pat_surname, pat_firstname = str(self.dicom.PatientName).split('^')
        except ValueError:
            pat_surname = str(self.dicom.PatientName)
            pat_firstname = ''

        pat_name = ' '.join((pat_surname, pat_firstname.title()))
        pat_age = self.dicom.get('PatientAge', '').strip('Y')

        pat_id = self.dicom.PatientID
        pat_sex = self.dicom.PatientSex
        try:
            pat_bdate = datetime.strptime(
                self.dicom.PatientBirthDate, '%Y%m%d').strftime("%e %b %Y")
        except ValueError:
            pat_bdate = ""

        # Strip microseconds from acquisition date
        regexp = r"\.\d+$"
        acquisition_date_no_micro = re.sub(
            regexp, '', self.dicom.AcquisitionDateTime)

        try:
            acquisition_date = datetime.strftime(
                datetime.strptime(
                    acquisition_date_no_micro, '%Y%m%d%H%M%S'),
                '%d %b %Y %H:%M'
            )
        except ValueError:
            acquisition_date = ""

        info = "%s\n%s: %s\n%s: %s\n%s: %s (%s %s)\n%s: %s" % (
            pat_name,
            i18n.pat_id,
            pat_id,
            i18n.pat_sex,
            pat_sex,
            i18n.pat_bdate,
            pat_bdate,
            pat_age,
            i18n.pat_age,
            i18n.acquisition_date,
            acquisition_date
        )

        plt.figtext(0.08, 0.87, info, fontsize=8)

        plt.figtext(0.30, 0.87, self.legend(), fontsize=8)

        if interpretation:
            plt.figtext(0.45, 0.87, self.interpretation(), fontsize=8)

        info = "%s: %s s %s: %s Hz" % (
            i18n.duration, self.duration,
            i18n.sampling_frequency,
            self.sampling_frequency
        )

        plt.figtext(0.08, 0.025, info, fontsize=8)

        info = INSTITUTION
        if not info:
            info = self.dicom.get('InstitutionName', '')

        plt.figtext(0.38, 0.025, info, fontsize=8)

        # TODO: the lowpass filter 0.05-40 Hz will have to became a parameter
        info = "%s mm/s %s mm/mV 0.05-40 Hz" % (self.mm_s, self.mm_mv)
        plt.figtext(0.76, 0.025, info, fontsize=8)

    def save(self, outputfile=None, outformat=None):
        """Save the plot result either on a file or on a output buffer,
        depending on the input params.

        @param outputfile: the output filename.
        @param outformat: the ouput file format.
        """

        def _save(output):
            plt.savefig(
                output, dpi=300, format=outformat,
                orientation='landscape'
            )

        if outputfile:
            _save(outputfile)
        else:
            output = io.BytesIO()
            _save(output)
            return output.getvalue()

    def plot(self, layoutid, mm_mv):
        """Plot the ecg signals inside the plotting area.
        Possible layout choice are:
        * 12x1 (one signal per line)
        * 6x2 (6 rows 2 columns)
        * 3x4 (4 signal chunk per line)
        * 3x4_1 (4 signal chunk per line. on the last line
        is drawn a complete signal)
        * ... and much much more

        The general rule is that the layout list is formed
        by as much lists as the number of lines we want to draw into the
        plotting area, each one containing the number of the signal chunk
        we want to plot in that line.

        @param layoutid: the desired layout
        @type layoutid: C{list} of C{list}
        """

        self.mm_mv = mm_mv

        layout = LAYOUT[layoutid]
        rows = len(layout)

        for numrow, row in enumerate(layout):

            columns = len(row)
            row_height = self.height / rows

            # Horizontal shift for lead labels and separators
            h_delta = self.samples / columns

            # Vertical shift of the origin
            v_delta = round(
                self.height * (1.0 - 1.0 / (rows * 2)) -
                numrow * (self.height / rows)
            )

            # Let's shift the origin on a multiple of 5 mm
            v_delta = (v_delta + 2.5) - (v_delta + 2.5) % 5

            # Lenght of a signal chunk
            chunk_size = int(self.samples / len(row))
            for numcol, signum in enumerate(row):
                left = numcol * chunk_size
                right = (1 + numcol) * chunk_size

                # The signal chunk, vertical shifted and
                # scaled by mm/mV factor
                signal = v_delta + mm_mv * self.signals[signum][left:right]
                self.axis.plot(
                    list(range(left, right)),
                    signal,
                    clip_on=False,
                    linewidth=0.6,
                    color='black',
                    zorder=10)
                if self.channel_definitions:
                    cseq = self.channel_definitions[signum].ChannelSourceSequence
                    meaning = cseq[0].CodeMeaning.replace(
                        'Lead', '').replace('(Einthoven)', '')
                elif self.channel_names:
                    meaning = self.channel_names[signum]
                else:
                    # Is there is no name, we print no name
                    meaning = ['']

                h = h_delta * numcol
                v = v_delta + row_height / 2.6
                plt.plot(
                    [h, h],
                    [v - 3, v],
                    lw=.6,
                    color='black',
                    zorder=50
                )

                self.axis.text(
                    h + 40,
                    v_delta + row_height / 3,
                    meaning,
                    zorder=50,
                    fontsize=8
                )

        # A4 size in inches
        self.fig.set_size_inches(11.69, 8.27)

    def draw(self,
             layoutid,
             mm_mv=10.0,
             minor_axis=True,
             interpretation=False,
             printinfo=None,
             border=False):
        """
        Draw grid and signals.  Always resets to a fresh figure/axis first.

        Parameters
        ----------
        layoutid : str
            One of the LAYOUT keys, e.g. "3x4_1".
        mm_mv : float
            Vertical scaling (mm per mV).
        minor_axis : bool
            Whether to draw minor gridlines.
        interpretation : bool
            Whether to render interpretation text.
        printinfo : bool or None
            If not None, overrides self.info_print.
        border : bool
            Pass‐through to create_figure(border=...).  
            If False, removes white margins; if True, uses default margins.
        """
        # 1) Optionally override info_print
        if printinfo is not None:
            self.info_print = printinfo

        # 2) Re‐create a brand‐new figure & axis on every draw call
        self.fig, self.axis = self.create_figure(border=border)

        # 3) Now draw the grid and 12‐lead paper trace
        self.draw_grid(minor_axis)
        self.plot(layoutid, mm_mv)

        # 4) Add patient‐info text if requested
        if self.info_print:
            self.print_info(interpretation)

        return self.fig


class ECGReadFileError(dicom.filereader.InvalidDicomError):
    pass


def extract_ecg_signal(dicom_path: str) -> np.ndarray:
    """Extract a 12-lead ECG signal from a DICOM file and return a (12, 5000) array in µV."""

    def butter_lowpass_filter(data, highcut, sampfreq, order=2):
        nyquist = 0.5 * sampfreq
        high = highcut / nyquist
        b, a = butter(order, high, btype='low')
        return lfilter(b, a, data)

    # Load DICOM
    with open(dicom_path, 'rb') as f:
        dicom_bytes = BytesIO(f.read())
    ds = dicom.dcmread(dicom_bytes)

    wf = ds.WaveformSequence[0]
    assert wf.WaveformSampleInterpretation == 'SS'
    assert wf.WaveformBitsAllocated == 16

    # Metadata
    channels = wf.NumberOfWaveformChannels
    samples = wf.NumberOfWaveformSamples
    sfreq = wf.SamplingFrequency
    raw = wf.WaveformData

    # Channel scaling
    scale_factor = np.ones(channels)
    offset = np.zeros(channels)
    units = []

    for i, ch in enumerate(wf.ChannelDefinitionSequence):
        if 'ChannelSensitivity' in ch:
            scale_factor[i] = float(ch.ChannelSensitivity) * float(ch.ChannelSensitivityCorrectionFactor)
        if 'ChannelBaseline' in ch:
            offset[i] = float(ch.ChannelBaseline)
        units.append(ch.ChannelSensitivityUnitsSequence[0].CodeValue)

    # Unpack and reshape
    data = np.asarray(struct.unpack('<{}h'.format(len(raw) // 2), raw), dtype=np.float32)
    signal = data.reshape((samples, channels)).T

    # Scale and filter to µV
    for i in range(channels):
        signal[i] = (signal[i] + offset[i]) * scale_factor[i]
        if units[i] == 'mV':
            signal[i] *= 1000.0  # Convert mV → µV
        elif units[i] == 'uV':
            pass  # Already in µV
        else:
            print(f"⚠️ Unknown unit '{units[i]}' for channel {i}, assuming µV.")

        # Remove baseline with median filter
        baseline = median_filter(signal[i], size=int(sfreq * 0.6))  # 600 ms window
        signal[i] = signal[i] - baseline

        # Low-pass filter
        signal[i] = butter_lowpass_filter(signal[i], 40.0, sfreq)

    # Trim flat tail (only check first lead)
    if signal.shape[1] >= 1000:
        tail = signal[0, -950:]
        if np.max(np.abs(tail)) < 5:  # flat if max amplitude < 5 µV
            print("Flat tail detected. Removing last 1000 samples.")
            signal = signal[:, :-1000]

    # Resample to 500 Hz → 5000 samples
    signal = resample(signal, 5000, axis=1)

    assert signal.shape == (12, 5000), f"Unexpected shape: {signal.shape}"
    return signal


def process_ecg_plot_from_dicom(dicom_path: str, fid: str, save_path: str):
    """
    Load, process, and plot a 12-lead ECG from a DICOM file and save it as an image.

    Args:
        dicom_path (str): Path to the DICOM file.
        fid (str): Unique identifier for output filename.
        save_path (str): Full path (without extension) to save the image.
    """
    # Load ECG signal
    try:
        signal = extract_ecg_signal(dicom_path)
    except Exception as e:
        raise RuntimeError(f"Failed to extract ECG from {dicom_path}: {e}")

    # Validate shape
    if signal.shape != (12, 5000):
        raise ValueError(f"Invalid ECG shape from {fid}: expected (12, 5000), got {signal.shape}")

    # Preprocess
    proc_signal = signal / 1000
    proc_signal2 = proc_signal - median_filter(proc_signal, size=(500, 1))
    full = proc_signal2.T  # (5000, 12)

    # Layout
    col1 = full[0:1250, 0:3].T
    col1a = full[0:1250, 0:1].T
    col2 = full[1250:2500, 3:6].T
    col2a = full[1250:2500, 0:1].T
    col3 = full[2500:3750, 6:9].T
    col3a = full[2500:3750, 0:1].T
    col4 = full[3750:5000, 9:12].T
    col4a = full[3750:5000, 0:1].T

    newplot1a = np.vstack((col1, col1a, col2, col2a, col3, col3a, col4, col4a))

    # Plot
    print(newplot1a)
    custom_ecg_plot(
        newplot1a,
        sample_rate=500,
        title="",
        show_separate_line=True,
        columns=4,
        lead_index=['I', 'II', 'III', 'I', 'aVR', 'aVL', 'aVF', '', 'V1', 'V2', 'V3', '', 'V4', 'V5', 'V6', ''],
        style='bw_alt'
    )

    # Clean style
    plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False,
                    right=False, labelleft=False, labelbottom=False)

    # Ensure directory exists and save as PNG
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.ioff()
    plt.savefig(os.path.join(save_path,  f'{fid}.png'), dpi=100, bbox_inches='tight')
    plt.close()
    gc.collect()
    
def process_ecg_plot_from_signal(signal, fid: str, save_path: str):
    """
    Load, process, and plot a 12-lead ECG from a DICOM file and save it as an image.

    Args:
        dicom_path (str): Path to the DICOM file.
        fid (str): Unique identifier for output filename.
        save_path (str): Full path (without extension) to save the image.
    """

    lead_index=['I', 'II', 'III', 'I','aVR', 'aVL', 'aVF','', 'V1', 'V2', 'V3', '','V4', 'V5', 'V6','']

    # Validate shape
    if signal.shape != (12, 5000):
        raise ValueError(f"Invalid ECG shape from {fid}: expected (12, 5000), got {signal.shape}")

    # Preprocess
    proc_signal = signal / 1000
    proc_signal2 = proc_signal - median_filter(proc_signal, size=(500, 1))
    full = signal.T  # (5000, 12)

    # Layout
    col1 = full[0:1250, 0:3].T
    col1a = full[0:1250, 0:1].T
    col2 = full[1250:2500, 3:6].T
    col2a = full[1250:2500, 0:1].T
    col3 = full[2500:3750, 6:9].T
    col3a = full[2500:3750, 0:1].T
    col4 = full[3750:5000, 9:12].T
    col4a = full[3750:5000, 0:1].T

    newplot1a = np.vstack((col1, col1a, col2, col2a, col3, col3a, col4, col4a))

    fig = custom_ecg_plot(
        newplot1a,
        sample_rate=500,
        title="",
        show_separate_line=True,
        columns=4,
        lead_index=['I', 'II', 'III', 'I', 'aVR', 'aVL', 'aVF', '', 'V1', 'V2', 'V3', '', 'V4', 'V5', 'V6', ''],
        style='bw_alt'
    )

    ax = fig.axes[0]
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False,
                    right=False, labelleft=False, labelbottom=False)

    # Ensure directory exists and save as PNG
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    

def custom_ecg_plot(
        ecg, 
        sample_rate    = 500, 
        title          = 'ECG 12', 
        lead_index     = None, 
        lead_order     = None,
        style          = None,
        columns        = 2,
        row_height     = 6,
        show_lead_name = True,
        show_grid      = True,
        show_separate_line = True,
        debug=False
    ):
    """Plot multi-lead ECG chart."""

    lead_index=['I', 'II', 'III', 'I','aVR', 'aVL', 'aVF','', 'V1', 'V2', 'V3', '','V4', 'V5', 'V6','']

    # --- Diagnostics and checks ---
    assert isinstance(ecg, np.ndarray), "ECG input must be a NumPy array"
    assert ecg.ndim == 2, f"ECG must be 2D (leads x samples), got shape {ecg.shape}"
    m, n = ecg.shape
    assert m >= 1 and n >= 100, f"ECG must have at least 1 lead and 100 samples, got shape {ecg.shape}"

    if lead_index is None:
        lead_index = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6'][:m]
    assert len(lead_index) == m, f"lead_index must have {m} names, but got {len(lead_index)}"

    if lead_order is None:
        lead_order = list(range(m))
    assert max(lead_order) < m, "lead_order contains invalid lead index"

    # --- Layout ---
    secs  = n / sample_rate
    leads = len(lead_order)
    rows  = int(ceil(leads / columns))
    display_factor = 1
    line_width = 0.5

    fig, ax = plt.subplots(figsize=(secs * columns * display_factor, rows * row_height / 5 * display_factor))
    display_factor = display_factor ** 0.5
    fig.subplots_adjust(hspace=0, wspace=0, left=0, right=1, bottom=0, top=1)
    fig.suptitle(title)

    x_min = 0
    x_max = columns * secs
    y_min = row_height / 4 - (rows / 2) * row_height
    y_max = row_height / 4

    # --- Grid styling ---
    color_schemes = {
        'bw':       ((0.4,0.4,0.4), (0.75,0.75,0.75), (0,0,0)),
        'bw_alt':   ((.6,.6,.6),    (0.9,0.9,0.9),     (0,0,0)),
        'black_pink': ((.65,.65,.65), (1,0.7,0.7),     (0,0,0)),
        'blue_pink': ((1,0,0),      (1,0.7,0.7),       (0,0,0.7)),
        None:       ((1,0,0),       (1,0.7,0.7),       (0,0,0.7))
    }
    color_major, color_minor, color_line = color_schemes.get(style, color_schemes[None])

    if show_grid:
        ax.set_xticks(np.arange(x_min, x_max, 0.2))    
        ax.set_yticks(np.arange(y_min, y_max, 0.5))
        ax.minorticks_on()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.grid(which='major', linestyle='-', linewidth=0.5 * display_factor, color=color_major)
        ax.grid(which='minor', linestyle='-', linewidth=0.5 * display_factor, color=color_minor)

    ax.set_ylim(y_min, y_max)
    ax.set_xlim(x_min, x_max)

    # --- Plot each lead ---
    for c in range(columns):
        for i in range(rows):
            lead_idx = c * rows + i
            if lead_idx >= leads:
                continue

            t_lead = lead_order[lead_idx]
            y_offset = -(row_height / 2) * ceil(i % rows)
            x_offset = secs * c if c > 0 else 0

            if debug:
                print(f"Plotting lead {t_lead}: {lead_index[t_lead]}")

            # Show vertical separator
            if show_separate_line and c > 0:
                try:
                    vline_y = ecg[t_lead][0] + y_offset
                    ax.plot([x_offset, x_offset], [vline_y - 0.3, vline_y + 0.3], linewidth=line_width * display_factor, color=color_line)
                except Exception as e:
                    print(f"Error drawing separator for lead {t_lead}: {e}")

            # Lead label
            if show_lead_name:
                ax.text(x_offset + 0.07, y_offset - 0.5, lead_index[t_lead], fontsize=9 * display_factor)

            # Plot ECG signal
            try:
                step = 1.0 / sample_rate
                x_vals = np.linspace(0, n / sample_rate, num=n, endpoint=False) + x_offset
                y_vals = ecg[t_lead] + y_offset

                if np.all(y_vals == 0):
                    print(f"⚠️  Lead {t_lead} ('{lead_index[t_lead]}') is flat (all zeros)")

                ax.plot(x_vals, y_vals, linewidth=line_width * display_factor, color=color_line)

            except Exception as e:
                print(f"❌ Error plotting lead {t_lead} ('{lead_index[t_lead]}'): {e}")

    return fig