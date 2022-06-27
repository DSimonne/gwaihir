import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import operator as operator_lib
from datetime import datetime
import tables as tb
import h5py
import shutil
from numpy.fft import fftshift
from scipy.ndimage import center_of_mass
from shlex import quote
from IPython.display import display

# PyNX
try:
    from pynx.cdi import CDI
    from pynx.cdi.runner.id01 import params
    from pynx.utils.math import smaller_primes
    pynx_import = True
except ModuleNotFoundError:
    pynx_import = False

# gwaihir package
import gwaihir

# bcdi package
from bcdi.preprocessing import ReadNxs3 as rd
from bcdi.utils.utilities import bin_data


def initialize_phase_retrieval(
    self,
    unused_label_data,
    parent_folder,
    iobs,
    mask,
    support,
    obj,
    auto_center_resize,
    max_size,
    unused_label_support,
    support_threshold,
    support_only_shrink,
    support_update_period,
    support_smooth_width,
    support_post_expand,
    unused_label_psf,
    psf,
    psf_model,
    fwhm,
    eta,
    update_psf,
    unused_label_algo,
    nb_hio,
    nb_raar,
    nb_er,
    nb_ml,
    nb_run,
    unused_label_filtering,
    filter_criteria,
    nb_run_keep,
    unused_label_options,
    live_plot,
    # zero_mask TODO,
    # crop_output TODO,
    positivity,
    beta,
    detwin,
    rebin,
    verbose,
    pixel_size_detector,
    unused_label_phase_retrieval,
    run_phase_retrieval,
    unused_label_run_pynx_tools,
    run_pynx_tools,
):
    """
    Get parameters values from widgets and run phase retrieval Possible
    to run phase retrieval via the CLI (with ot without MPI) Or directly in
    python using the operators.

    :param parent_folder: folder in which the raw data files are, and where the
     output will be saved
    :param iobs: 2D/3D observed diffraction data (intensity).
      Assumed to be corrected and following Poisson statistics, will be
      converted to float32. Dimensions should be divisible by 4 and have a
      prime factor decomposition up to 7. Internally, the following special
      values are used:
      * values<=-1e19 are masked. Among those, values in ]-1e38;-1e19] are
         estimated values, stored as -(iobs_est+1)*1e19, which can be used
         to make a loose amplitude projection.
        Values <=-1e38 are masked (no amplitude projection applied), just
        below the minimum float32 value
      * -1e19 < values <= 1 are observed but used as free pixels
        If the mask is not supplied, then it is assumed that the above
        special values are used.
    :param support: initial support in real space (1 = inside support,
     0 = outside)
    :param obj: initial object. If None, it should be initialised later.
    :param mask: mask for the diffraction data (0: valid pixel, >0: masked)
    :param auto_center_resize: if used (command-line keyword) or =True,
     the input data will be centered and cropped  so that the size of the
     array is compatible with the (GPU) FFT library used. If 'roi' is used,
     centering is based on ROI. [default=False]
    :param max_size=256: maximum size for the array used for analysis,
     along all dimensions. The data will be cropped to this value after
     centering. [default: no maximum size]
    :param support_threshold: must be between 0 and 1. Only points with
     object amplitude above a value equal to relative_threshold *
     reference_value are kept in the support.
     reference_value can either:
        - use the fact that when converged, the square norm of the object
        is equal to the number of recorded photons (normalized Fourier
        Transform). Then: reference_value = sqrt((abs(obj)**2).sum()/
        nb_points_support)
        - or use threshold_percentile (see below, very slow, deprecated)
    :param support_smooth_width: smooth the object amplitude using a
     gaussian of this width before calculating new support
     If this is a scalar, the smooth width is fixed to this value.
     If this is a 3-value tuple (or list or array), i.e. 'smooth_width=2,
     0.5,600', the smooth width will vary with the number of cycles
     recorded in the CDI object (as cdi.cycle), varying exponentially from
     the first to the second value over the number of cycles specified by
     the last value.
     With 'smooth_width=a,b,nb':
     - smooth_width = a * exp(-cdi.cycle/nb*log(b/a)) if cdi.cycle < nb
     - smooth_width = b if cdi.cycle >= nb
    :param support_only_shrink: if True, the support can only shrink
    :param method: either 'max' or 'average' or 'rms' (default), the
     threshold will be relative to either the maximum amplitude in the
     object, or the average or root-mean-square amplitude (computed inside
     support)
    :param support_post_expand=1: after the new support has been calculated,
    it can be processed using the SupportExpand operator, either one or
    multiple times, in order to 'clean' the support:
     - 'post_expand=1' will expand the support by 1 pixel
     - 'post_expand=-1' will shrink the support by 1 pixel
     - 'post_expand=(-1,1)' will shrink and then expand the support by
     1 pixel
     - 'post_expand=(-2,3)' will shrink and then expand the support by
     respectively 2 and 3 pixels
    :param psf: e.g. True
     whether or not to use the PSF, partial coherence point-spread function,
     estimated with 50 cycles of Richardson-Lucy
    :param psf_model: "lorentzian", "gaussian" or "pseudo-voigt", or None
     to deactivate
    :param fwhm: the full-width at half maximum, in pixels
    :param eta: the eta parameter for the pseudo-voigt
    :param update_psf: how often the psf is updated
    :param nb_raar: number of relaxed averaged alternating reflections
     cycles, which the algorithm will use first. During RAAR and HIO, the
     support is updated regularly
    :param nb_hio: number of hybrid input/output cycles, which the
     algorithm will use after RAAR. During RAAR and HIO, the support is
     updated regularly
    :param nb_er: number of error reduction cycles, performed after HIO,
     without support update
    :param nb_ml: number of maximum-likelihood conjugate gradient to
     perform after ER
    :param nb_run: number of times to run the optimization
    :param nb_run_keep: number of best run results to keep, according to
     filter_criteria.
    :param filter_criteria: e.g. "LLK"
        criteria onto which the best solutions will be chosen
    :param live_plot: a live plot will be displayed every N cycle
    :param beta: the beta value for the HIO operator
    :param positivity: True or False
    :param zero_mask: if True, masked pixels (iobs<-1e19) are forced to
     zero, otherwise the calculated complex amplitude is kept with an
     optional scale factor.
    :param detwin: if set (command-line) or if detwin=True (parameters
     file), 10 cycles will be performed at 25% of the total number of
     RAAR or HIO cycles, with a support cut in half to bias towards one
     twin image
    :param pixel_size_detector: detector pixel size (meters)
    :param wavelength: experiment wavelength (meters)
    :param detector_distance: detector distance (meters)
    """
    # Assign attributes
    self.Dataset.parent_folder = parent_folder
    self.Dataset.iobs = parent_folder + iobs
    if mask != "":
        self.Dataset.mask = parent_folder + mask
    else:
        self.Dataset.mask = ""
    if support != "":
        self.Dataset.support = parent_folder + support
    else:
        self.Dataset.support = ""
    if obj != "":
        self.Dataset.obj = parent_folder + obj
    else:
        self.Dataset.obj = ""
    self.Dataset.auto_center_resize = auto_center_resize
    self.Dataset.max_size = max_size
    self.Dataset.support_threshold = support_threshold
    self.Dataset.support_only_shrink = support_only_shrink
    self.Dataset.support_update_period = support_update_period
    self.Dataset.support_smooth_width = support_smooth_width
    self.Dataset.support_post_expand = support_post_expand
    self.Dataset.psf = psf
    self.Dataset.psf_model = psf_model
    self.Dataset.fwhm = fwhm
    self.Dataset.eta = eta
    self.Dataset.update_psf = update_psf
    self.Dataset.nb_raar = nb_raar
    self.Dataset.nb_hio = nb_hio
    self.Dataset.nb_er = nb_er
    self.Dataset.nb_ml = nb_ml
    self.Dataset.nb_run = nb_run
    self.Dataset.filter_criteria = filter_criteria
    self.Dataset.nb_run_keep = nb_run_keep
    self.Dataset.live_plot = live_plot
    # To do
    # self.Dataset.zero_mask = zero_mask
    # self.Dataset.crop_output = crop_output
    self.Dataset.positivity = positivity
    self.Dataset.beta = beta
    self.Dataset.detwin = detwin
    self.Dataset.rebin = rebin
    self.Dataset.verbose = verbose
    self.Dataset.pixel_size_detector = np.round(
        pixel_size_detector * 1e-6, 6)
    self.run_phase_retrieval = run_phase_retrieval
    self.run_pynx_tools = run_pynx_tools

    # Extract dict, list and tuple from strings
    self.Dataset.support_threshold = literal_eval(
        self.Dataset.support_threshold)
    self.Dataset.support_smooth_width = literal_eval(
        self.Dataset.support_smooth_width)
    self.Dataset.support_post_expand = literal_eval(
        self.Dataset.support_post_expand)
    self.Dataset.rebin = literal_eval(self.Dataset.rebin)

    if self.Dataset.live_plot == 0:
        self.Dataset.live_plot = False

    print("Scan n°", self.Dataset.scan)

    self.Dataset.energy = self._list_widgets_preprocessing.children[50].value
    self.Dataset.wavelength = 1.2399 * 1e-6 / self.Dataset.energy
    self.Dataset.detector_distance = self._list_widgets_preprocessing.children[49].value

    print("\tCXI input: Energy = %8.2f eV" % self.Dataset.energy)
    print(f"\tCXI input: Wavelength = {self.Dataset.wavelength*1e10} A")
    print("\tCXI input: detector distance = %8.2f m" %
          self.Dataset.detector_distance)
    print(
        f"\tCXI input: detector pixel size = {self.Dataset.pixel_size_detector} m")

    # PyNX arguments text files
    self.Dataset.pynx_parameter_gui_file = self.preprocessing_folder\
        + "/pynx_run_gui.txt"
    self.Dataset.pynx_parameter_cli_file = self.preprocessing_folder\
        + "/pynx_run.txt"

    # Phase retrieval
    if self.run_phase_retrieval and not self.run_pynx_tools:
        if self.run_phase_retrieval in ("batch", "local_script"):
            # Create /gui_run/ directory
            try:
                os.mkdir(
                    f"{self.preprocessing_folder}/gui_run/")
                print(
                    f"\tCreated {self.preprocessing_folder}/gui_run/", end="\n\n")
            except (FileExistsError, PermissionError):
                print(
                    f"{self.preprocessing_folder}/gui_run/ exists", end="\n\n")

            self.text_file = []
            self.Dataset.live_plot = False

            # Load files
            self.text_file.append("# Parameters\n")
            for file, parameter in [
                    (self.Dataset.iobs, "data"),
                    (self.Dataset.mask, "mask"),
                    (self.Dataset.obj, "object")
            ]:
                if file != "":
                    self.text_file.append(f"{parameter} = \"{file}\"\n")

            if support != "":
                self.text_file += [
                    f"support = \"{self.Dataset.support}\"\n",
                    '\n']
            # else no support, just don't write it

            # Clean threshold syntax
            support_threshold = support_threshold.replace("(", "")
            support_threshold = support_threshold.replace(")", "")
            support_threshold = support_threshold.replace(" ", "")

            # Other support parameters
            self.text_file += [
                f'support_threshold= {support_threshold}\n',
                f'support_only_shrink = {self.Dataset.support_only_shrink}\n',
                f'support_update_period = {self.Dataset.support_update_period}\n',
                f'support_smooth_width_begin = {self.Dataset.support_smooth_width[0]}\n',
                f'support_smooth_width_end = {self.Dataset.support_smooth_width[1]}\n',
                f'support_post_expand = {self.Dataset.support_post_expand}\n'
                '\n',
            ]

            # PSF
            if self.Dataset.psf:
                if self.Dataset.psf_model != "pseudo-voigt":
                    self.text_file.append(
                        f"psf = \"{self.Dataset.psf_model},{self.Dataset.fwhm}\"\n")

                if self.Dataset.psf_model == "pseudo-voigt":
                    self.text_file.append(
                        f"psf = \"{self.Dataset.psf_model},{self.Dataset.fwhm},{self.Dataset.eta}\"\n")
            # no PSF, just don't write anything

            # Filtering the reconstructions
            if self.Dataset.filter_criteria == "LLK":
                nb_run_keep_LLK = self.Dataset.nb_run_keep
                nb_run_keep_std = False

            elif self.Dataset.filter_criteria == "std":
                nb_run_keep_LLK = self.Dataset.nb_run
                nb_run_keep_std = self.Dataset.nb_run_keep

            elif self.Dataset.filter_criteria == "LLK_standard_deviation":
                nb_run_keep_LLK = self.Dataset.nb_run_keep + \
                    (self.Dataset.nb_run - self.Dataset.nb_run_keep) // 2
                nb_run_keep_std = self.Dataset.nb_run_keep

            # Clean rebin syntax
            rebin = rebin.replace("(", "")
            rebin = rebin.replace(")", "")
            rebin = rebin.replace(" ", "")

            # Other parameters
            self.text_file += [
                'data2cxi = True\n',
                f'auto_center_resize = {self.Dataset.auto_center_resize}\n',
                '\n',
                f'nb_raar = {self.Dataset.nb_raar}\n',
                f'nb_hio = {self.Dataset.nb_hio}\n',
                f'nb_er = {self.Dataset.nb_er}\n',
                f'nb_ml = {self.Dataset.nb_ml}\n',
                '\n',
                f'nb_run = {self.Dataset.nb_run}\n',
                f'nb_run_keep = {nb_run_keep_LLK}\n',
                '\n',
                f'# max_size = {self.Dataset.max_size}\n',
                'zero_mask = auto # masked pixels will start from imposed 0 and then let free\n',
                'crop_output= 0 # set to 0 to avoid cropping the output in the .cxi\n',
                "mask_interp=8,2\n"
                "confidence_interval_factor_mask=0.5,1.2\n"
                '\n',
                f'positivity = {self.Dataset.positivity}\n',
                f'beta = {self.Dataset.beta}\n',
                f'detwin = {self.Dataset.detwin}\n',
                f'rebin = {rebin}\n',
                '\n',
                '# Generic parameters\n',
                f'detector_distance = {self.Dataset.detector_distance}\n',
                f'pixel_size_detector = {self.Dataset.pixel_size_detector}\n',
                f'wavelength = {self.Dataset.wavelength}\n',
                f'verbose = {self.Dataset.verbose}\n',
                "output_format= 'cxi'\n",
                f'live_plot = {self.Dataset.live_plot}\n',
                "mpi=run\n",
            ]

            with open(self.Dataset.pynx_parameter_gui_file, "w") as v:
                for line in self.text_file:
                    v.write(line)

            gutil.hash_print(
                f"Saved parameters in: {self.Dataset.pynx_parameter_gui_file}")

            if self.run_phase_retrieval == "batch":
                # Runs modes directly and saves all data in a "gui_run"
                # subdir, filter based on LLK
                print(
                    f"\nRunning: $ {self.path_scripts}/run_slurm_job.sh "
                    f"--reconstruct gui --username {self.user_name} "
                    f"--path {self.preprocessing_folder} "
                    f"--filtering {nb_run_keep_std} --modes true"
                )
                print(
                    "\nSolution filtering and modes decomposition are "
                    "automatically applied at the end of the batch job."
                )
                os.system(
                    "{}/run_slurm_job.sh \
                    --reconstruct gui \
                    --username {} \
                    --path {} \
                    --filtering {} \
                    --modes true".format(
                        quote(self.path_scripts),
                        quote(self.user_name),
                        quote(self.preprocessing_folder),
                        quote(str(nb_run_keep_std)),
                    )
                )

                # Copy Pynx parameter file in folder
                shutil.copyfile(self.Dataset.pynx_parameter_gui_file,
                                f"{self.preprocessing_folder}/gui_run/pynx_run_gui.txt")

            elif self.run_phase_retrieval == "local_script":
                try:
                    print(
                        f"\nRunning: $ {self.path_scripts}/pynx-id01cdi.py "
                        "pynx_run_gui.txt 2>&1 | tee README_pynx_local_script.md &",
                        end="\n\n")
                    os.system(
                        "cd {}; {}/pynx-id01cdi.py pynx_run_gui.txt 2>&1 | tee README_pynx_local_script.md &".format(
                            quote(self.preprocessing_folder),
                            quote(self.path_scripts),
                        )
                    )
                except KeyboardInterrupt:
                    print("Phase retrieval stopped by user ...")

        elif self.run_phase_retrieval == "operators":
            # Extract data
            gutil.hash_print(
                "Log likelihood is updated every 50 iterations.")
            self.Dataset.calc_llk = 50  # TODO

            # Keep a list of the resulting scans
            self.reconstruction_file_list = []

            try:
                # Initialise the cdi operator
                raw_cdi = gutil.initialize_cdi_operator(
                    iobs=self.Dataset.iobs,
                    mask=self.Dataset.mask,
                    support=self.Dataset.support,
                    obj=self.Dataset.obj,
                    rebin=self.Dataset.rebin,
                    auto_center_resize=self.Dataset.auto_center_resize,
                    max_size=self.Dataset.max_size,
                    wavelength=self.Dataset.wavelength,
                    pixel_size_detector=self.Dataset.pixel_size_detector,
                    detector_distance=self.Dataset.detector_distance,
                )

                # Run phase retrieval for nb_run
                for i in range(self.Dataset.nb_run):
                    print(
                        "\n###########################################"
                        "#############################################"
                        f"\nRun {i}"
                    )

                    # Make a copy to gain time
                    cdi = raw_cdi.copy()

                    # Save instance
                    if i == 0:
                        cxi_filename = "{}/preprocessing/{}.cxi".format(
                            self.Dataset.scan_folder,
                            self.Dataset.iobs.split("/")[-1].split(".")[0]
                        )

                        gutil.save_cdi_operator_as_cxi(
                            gwaihir_dataset=self.Dataset,
                            cdi_operator=cdi,
                            path_to_cxi=cxi_filename,
                        )

                    if i > 4:
                        print("Stopping liveplot to go faster\n")
                        self.Dataset.live_plot = False

                    # Change support threshold for supports update
                    if isinstance(self.Dataset.support_threshold, float):
                        self.Dataset.threshold_relative\
                            = self.Dataset.support_threshold
                    elif isinstance(self.Dataset.support_threshold, tuple):
                        self.Dataset.threshold_relative = np.random.uniform(
                            self.Dataset.support_threshold[0],
                            self.Dataset.support_threshold[1]
                        )
                    print(f"Threshold: {self.Dataset.threshold_relative}")

                    # Create support object
                    sup = SupportUpdate(
                        threshold_relative=self.Dataset.threshold_relative,
                        smooth_width=self.Dataset.support_smooth_width,
                        force_shrink=self.Dataset.support_only_shrink,
                        method='rms',
                        post_expand=self.Dataset.support_post_expand,
                    )

                    # Initialize the free pixels for LLK
                    # cdi = InitFreePixels() * cdi

                    # Initialize the support with autocorrelation, if no
                    # support given
                    if not self.Dataset.support:
                        sup_init = "autocorrelation"
                        if isinstance(self.Dataset.live_plot, int):
                            if i > 4:
                                cdi = ScaleObj() * AutoCorrelationSupport(
                                    threshold=0.1,  # extra argument
                                    verbose=True) * cdi

                            else:
                                cdi = ShowCDI() * ScaleObj() \
                                    * AutoCorrelationSupport(
                                    threshold=0.1,  # extra argument
                                    verbose=True) * cdi

                        else:
                            cdi = ScaleObj() * AutoCorrelationSupport(
                                threshold=0.1,  # extra argument
                                verbose=True) * cdi
                    else:
                        sup_init = "support"

                    # Begin with HIO cycles without PSF and with support
                    # updates
                    try:
                        # update_psf = 0 probably enough but not sure
                        if self.Dataset.psf:
                            if self.Dataset.support_update_period == 0:
                                cdi = HIO(
                                    beta=self.Dataset.beta,
                                    calc_llk=self.Dataset.calc_llk,
                                    show_cdi=self.Dataset.live_plot
                                ) ** self.Dataset.nb_hio * cdi
                                cdi = RAAR(
                                    beta=self.Dataset.beta,
                                    calc_llk=self.Dataset.calc_llk,
                                    show_cdi=self.Dataset.live_plot
                                ) ** (self.Dataset.nb_raar // 2) * cdi

                                # PSF is introduced at 66% of HIO and RAAR
                                if psf_model != "pseudo-voigt":
                                    cdi = InitPSF(
                                        model=self.Dataset.psf_model,
                                        fwhm=self.Dataset.fwhm,
                                    ) * cdi

                                elif psf_model == "pseudo-voigt":
                                    cdi = InitPSF(
                                        model=self.Dataset.psf_model,
                                        fwhm=self.Dataset.fwhm,
                                        eta=self.Dataset.eta,
                                    ) * cdi

                                cdi = RAAR(
                                    beta=self.Dataset.beta,
                                    calc_llk=self.Dataset.calc_llk,
                                    show_cdi=self.Dataset.live_plot,
                                    update_psf=self.Dataset.update_psf
                                ) ** (self.Dataset.nb_raar // 2) * cdi
                                cdi = ER(
                                    calc_llk=self.Dataset.calc_llk,
                                    show_cdi=self.Dataset.live_plot,
                                    update_psf=self.Dataset.update_psf
                                ) ** self.Dataset.nb_er * cdi

                            else:
                                hio_power = self.Dataset.nb_hio \
                                    // self.Dataset.support_update_period
                                raar_power = (
                                    self.Dataset.nb_raar // 2) \
                                    // self.Dataset.support_update_period
                                er_power = self.Dataset.nb_er \
                                    // self.Dataset.support_update_period

                                cdi = (sup * HIO(
                                    beta=self.Dataset.beta,
                                    calc_llk=self.Dataset.calc_llk,
                                    show_cdi=self.Dataset.live_plot
                                )**self.Dataset.support_update_period
                                ) ** hio_power * cdi
                                cdi = (sup * RAAR(
                                    beta=self.Dataset.beta,
                                    calc_llk=self.Dataset.calc_llk,
                                    show_cdi=self.Dataset.live_plot
                                )**self.Dataset.support_update_period
                                ) ** raar_power * cdi

                                # PSF is introduced at 66% of HIO and RAAR
                                # so from cycle n°924
                                if psf_model != "pseudo-voigt":
                                    cdi = InitPSF(
                                        model=self.Dataset.psf_model,
                                        fwhm=self.Dataset.fwhm,
                                    ) * cdi

                                elif psf_model == "pseudo-voigt":
                                    cdi = InitPSF(
                                        model=self.Dataset.psf_model,
                                        fwhm=self.Dataset.fwhm,
                                        eta=self.Dataset.eta,
                                    ) * cdi

                                cdi = (sup * RAAR(
                                    beta=self.Dataset.beta,
                                    calc_llk=self.Dataset.calc_llk,
                                    show_cdi=self.Dataset.live_plot,
                                    update_psf=self.Dataset.update_psf
                                )**self.Dataset.support_update_period
                                ) ** raar_power * cdi
                                cdi = (sup * ER(
                                    calc_llk=self.Dataset.calc_llk,
                                    show_cdi=self.Dataset.live_plot,
                                    update_psf=self.Dataset.update_psf
                                )**self.Dataset.support_update_period
                                ) ** er_power * cdi

                        if not self.Dataset.psf:
                            if self.Dataset.support_update_period == 0:

                                cdi = HIO(
                                    beta=self.Dataset.beta,
                                    calc_llk=self.Dataset.calc_llk,
                                    show_cdi=self.Dataset.live_plot
                                ) ** self.Dataset.nb_hio * cdi
                                cdi = RAAR(
                                    beta=self.Dataset.beta,
                                    calc_llk=self.Dataset.calc_llk,
                                    show_cdi=self.Dataset.live_plot
                                ) ** self.Dataset.nb_raar * cdi
                                cdi = ER(
                                    calc_llk=self.Dataset.calc_llk,
                                    show_cdi=self.Dataset.live_plot
                                ) ** self.Dataset.nb_er * cdi

                            else:
                                hio_power = self.Dataset.nb_hio \
                                    // self.Dataset.support_update_period
                                raar_power = self.Dataset.nb_raar \
                                    // self.Dataset.support_update_period
                                er_power = self.Dataset.nb_er \
                                    // self.Dataset.support_update_period

                                cdi = (sup * HIO(
                                    beta=self.Dataset.beta,
                                    calc_llk=self.Dataset.calc_llk,
                                    show_cdi=self.Dataset.live_plot
                                )**self.Dataset.support_update_period
                                ) ** hio_power * cdi
                                cdi = (sup * RAAR(
                                    beta=self.Dataset.beta,
                                    calc_llk=self.Dataset.calc_llk,
                                    show_cdi=self.Dataset.live_plot
                                )**self.Dataset.support_update_period
                                ) ** raar_power * cdi
                                cdi = (sup * ER(
                                    calc_llk=self.Dataset.calc_llk,
                                    show_cdi=self.Dataset.live_plot
                                )**self.Dataset.support_update_period
                                ) ** er_power * cdi

                        fn = "{}/result_scan_{}_run_{}_LLK_{:.4}_support_threshold_{:.4}_shape_{}_{}_{}_{}.cxi".format(
                            self.Dataset.parent_folder,
                            self.Dataset.scan,
                            i,
                            cdi.get_llk()[0],
                            self.Dataset.threshold_relative,
                            cdi.iobs.shape[0],
                            cdi.iobs.shape[1],
                            cdi.iobs.shape[2],
                            sup_init,
                        )

                        self.reconstruction_file_list.append(fn)
                        cdi.save_obj_cxi(fn)
                        print(
                            f"\nSaved as {fn}."
                            "\n###########################################"
                            "#############################################"
                        )

                    except SupportTooLarge:
                        print(
                            "Threshold value probably too low, support too large too continue")

                # If filter, filter data
                if self.Dataset.filter_criteria:
                    gutil.filter_reconstructions(
                        self.Dataset.parent_folder,
                        self.Dataset.nb_run,
                        self.Dataset.nb_run_keep,
                        self.Dataset.filter_criteria
                    )

            except KeyboardInterrupt:
                clear_output(True)
                gutil.hash_print(
                    "Phase retrieval stopped by user, cxi file list below."
                )

            gutil.list_reconstructions(
                folder=self.preprocessing_folder,
                scan_name=self.Dataset.scan_name
            )

    # Modes decomposition and solution filtering
    if self.run_pynx_tools and not self.run_phase_retrieval:
        if self.run_pynx_tools == "modes":
            gutil.run_modes_decomposition(
                path_scripts=self.path_scripts,
                folder=self.Dataset.parent_folder,
            )

        elif self.run_pynx_tools == "filter":
            gutil.filter_reconstructions(
                folder=self.Dataset.parent_folder,
                nb_run=None,  # Will take the amount of cxi files found
                nb_run_keep=self.Dataset.nb_run_keep,
                filter_criteria=self.Dataset.filter_criteria
            )

    # Clean output
    if not self.run_phase_retrieval and not self.run_pynx_tools:
        gutil.hash_print("Cleared output.")
        clear_output(True)

        gutil.list_reconstructions(
            folder=self.preprocessing_folder,
            scan_name=self.Dataset.scan_name
        )

        # Refresh folders
        self.sub_directories_handler(change=self.Dataset.scan_folder)

        # Plot folder, refresh values
        self.tab_data.children[1].value = self.preprocessing_folder
        self.plot_folder_handler(change=self.preprocessing_folder)

        # Strain folder, refresh values
        self._list_widgets_strain.children[-4].value\
            = self.preprocessing_folder
        self.strain_folder_handler(change=self.preprocessing_folder)


def filter_reconstructions(
    folder,
    nb_run_keep,
    nb_run=None,
    filter_criteria="LLK"
):
    """
    Filter the phase retrieval output depending on a given parameter,
    for now only LLK and standard deviation are available. This allows the
    user to run a lot of reconstructions but to then automatically keep the
    "best" ones, according to this parameter. filter_criteria can take the
    values "LLK" or "standard_deviation" If you filter based on both, the
    function will filter nb_run_keep/2 files by the first criteria, and the
    remaining files by the second criteria.

    The parameters are specified in the phase retrieval tab, and
    their values saved through self.initialize_phase_retrieval()

    .param folder: parent folder to cxi files
    :param nb_run_keep: number of best run results to keep in the end,
     according to filter_criteria.
    :param nb_run: number of times to run the optimization, if None, equal
     to nb of files detected
    :param filter_criteria: default "LLK"
     criteria onto which the best solutions will be chosen
     possible values are ("standard_deviation", "LLK",
     "standard_deviation_LLK", "LLK_standard_deviation")
    """
    # Sorting functions depending on filtering criteria
    def filter_by_std(cxi_files, nb_run_keep):
        """Use the standard deviation of the reconstructed object as
        filtering criteria.

        The lowest standard deviations are best.
        """
        # Keep filtering criteria of reconstruction modules in dictionnary
        filtering_criteria_value = {}

        print(
            "\n###################"
            "#####################"
            "#####################"
            "#####################"
        )
        print("Computing standard deviation of object modulus for scans:")
        for filename in cxi_files:
            print(f"\t{os.path.basename(filename)}")
            with tb.open_file(filename, "r") as f:
                data = f.root.entry_1.image_1.data[:]
                amp = np.abs(data)
                # Skip values near 0
                meaningful_data = amp[amp > 0.05 * amp.max()]
                filtering_criteria_value[filename] = np.std(
                    meaningful_data
                )

        # Sort files
        sorted_dict = sorted(
            filtering_criteria_value.items(),
            key=operator_lib.itemgetter(1)
        )

        # Remove files
        print("\nRemoving scans:")
        for filename, filtering_criteria_value in sorted_dict[nb_run_keep:]:
            print(f"\t{os.path.basename(filename)}")
            os.remove(filename)
        print(
            "#####################"
            "#####################"
            "#####################"
            "###################\n"
        )

    def filter_by_LLK(cxi_files, nb_run_keep):
        """
        Use the free log-likelihood values of the reconstructed object
        as filtering criteria.

        The lowest standard deviations are best. See PyNX for
        details
        """
        # Keep filtering criteria of reconstruction modules in dictionnary
        filtering_criteria_value = {}

        print(
            "\n###################"
            "#####################"
            "#####################"
            "#####################"
        )
        print("Extracting LLK value (poisson statistics) for scans:")
        for filename in cxi_files:
            print(f"\t{os.path.basename(filename)}")
            with tb.open_file(filename, "r") as f:
                llk = f.root.entry_1.image_1.process_1.\
                    results.llk_poisson[...]
                filtering_criteria_value[filename] = llk

        # Sort files
        sorted_dict = sorted(
            filtering_criteria_value.items(),
            key=operator_lib.itemgetter(1)
        )

        # Remove files
        print("\nRemoving scans:")
        for filename, filtering_criteria_value in sorted_dict[nb_run_keep:]:
            print(f"\t{os.path.basename(filename)}")
            os.remove(filename)
        print(
            "#####################"
            "#####################"
            "#####################"
            "###################\n"
        )

    # Main function supporting different cases
    try:
        print(
            "\n###################"
            "#####################"
            "#####################"
            "#####################"
        )
        print("Iterating on files matching:")
        print(f"\t{folder}/*LLK*.cxi")
        cxi_files = sorted(glob.glob(f"{folder}/result_scan*LLK*.cxi"))
        print(
            "#####################"
            "#####################"
            "#####################"
            "###################\n"
        )

        if cxi_files == []:
            print(f"No *LLK*.cxi files in {folder}/result_scan*LLK*.cxi")

        else:
            # only standard_deviation
            if filter_criteria is "standard_deviation":
                filter_by_std(cxi_files, nb_run_keep)

            # only LLK
            elif filter_criteria is "LLK":
                filter_by_LLK(cxi_files, nb_run_keep)

            # standard_deviation then LLK
            elif filter_criteria is "standard_deviation_LLK":
                if nb_run is None:
                    nb_run = len(cxi_files)

                filter_by_std(cxi_files, nb_run_keep +
                              (nb_run - nb_run_keep) // 2)

                hash_print("Iterating on remaining files.")

                cxi_files = sorted(
                    glob.glob(f"{folder}/result_scan*LLK*.cxi"))

                if cxi_files == []:
                    print(
                        f"No *LLK*.cxi files remaining in \
                        {folder}/result_scan*LLK*.cxi")
                else:
                    filter_by_LLK(cxi_files, nb_run_keep)

            # LLK then standard_deviation
            elif filter_criteria is "LLK_standard_deviation":
                if nb_run is None:
                    nb_run = len(cxi_files)

                filter_by_LLK(cxi_files, nb_run_keep +
                              (nb_run - nb_run_keep) // 2)

                hash_print("Iterating on remaining files.")

                cxi_files = sorted(
                    glob.glob(f"{folder}/result_scan*LLK*.cxi"))

                if cxi_files == []:
                    print(
                        f"No *LLK*.cxi files remaining in \
                        {folder}/result_scan*LLK*.cxi")
                else:
                    filter_by_std(cxi_files, nb_run_keep)

            else:
                hash_print("No filtering")
    except KeyboardInterrupt:
        hash_print("File filtering stopped by user ...")


def initialize_cdi_operator(
    iobs,
    mask=None,
    support=None,
    obj=None,
    rebin=(1, 1, 1),
    auto_center_resize=False,
    max_size=None,
    wavelength=None,
    pixel_size_detector=None,
    detector_distance=None,
):
    """
    Initialize the cdi operator by processing the possible inputs:
        - iobs
        - mask
        - support
        - obj
    Will also crop and center the data if specified.

    :param iobs: path to npz or npy that stores this array
    :param mask: path to npz or npy that stores this array
    :param support: path to npz or npy that stores this array
    :param obj: path to npz or npy that stores this array
    :param rebin: e.g. (1, 1, 1), applied to all the arrays
    :param auto_center_resize:
    :param max_size:
    :param wavelength:
    :param pixel_size_detector:
    :param detector_distance:

    return: cdi operator
    """
    if iobs in ("", None) or not os.path.isfile(iobs):
        # Dataset.iobs = None
        iobs = None
        print("At least iobs must exist.")
        return None  # stop function directly

    if iobs.endswith(".npy"):
        iobs = np.load(iobs)
        print("\tCXI input: loading data")
    elif iobs.endswith(".npz"):
        try:
            iobs = np.load(iobs)["data"]
            print("\tCXI input: loading data")
        except KeyError:
            print("\t\"data\" key does not exist.")
            raise KeyboardInterrupt

    if rebin != (1, 1, 1):
        iobs = bin_data(iobs, rebin)

    # fft shift
    iobs = fftshift(iobs)

    if mask not in ("", None) or not os.path.isfile(mask):
        if mask.endswith(".npy"):
            mask = np.load(mask).astype(np.int8)
            nb = mask.sum()
            print("\tCXI input: loading mask, with %d pixels masked (%6.3f%%)" % (
                nb, nb * 100 / mask.size))
        elif mask.endswith(".npz"):
            try:
                mask = np.load(mask)[
                    "mask"].astype(np.int8)
                nb = mask.sum()
                print("\tCXI input: loading mask, with %d pixels masked (%6.3f%%)" % (
                    nb, nb * 100 / mask.size))
            except KeyError:
                print("\t\"mask\" key does not exist.")

        if rebin != (1, 1, 1):
            mask = bin_data(mask, rebin)

        # fft shift
        mask = fftshift(mask)

    else:
        mask = None

    print(support)
    if support not in ("", None) or not os.path.isfile(support):
        if support.endswith(".npy"):
            support = np.load(support)
            print("\tCXI input: loading support")
        elif support.endswith(".npz"):
            try:
                support = np.load(support)["data"]
                print("\tCXI input: loading support")
            except (FileNotFoundError, ValueError):
                print("\tFile not supported or does not exist.")
            except KeyError:
                print("\t\"data\" key does not exist.")
                try:
                    support = np.load(support)["support"]
                    print("\tCXI input: loading support")
                except KeyError:
                    print("\t\"support\" key does not exist.")
                    try:
                        support = np.load(support)["obj"]
                        print("\tCXI input: loading support")
                    except KeyError:
                        print(
                            "\t\"obj\" key does not exist."
                            "\t--> Could not load support array."
                        )

        if rebin != (1, 1, 1):
            support = bin_data(support, rebin)

        # fft shift
        try:
            support = fftshift(support)
        except ValueError:
            support = None

    else:
        support = None

    print(obj)
    if obj not in ("", None) or not os.path.isfile(obj):
        if obj.endswith(".npy"):
            obj = np.load(obj)
            print("\tCXI input: loading object")
        elif obj.endswith(".npz"):
            try:
                obj = np.load(obj)["data"]
                print("\tCXI input: loading object")
            except KeyError:
                print("\t\"data\" key does not exist.")

        if rebin != (1, 1, 1):
            obj = bin_data(obj, rebin)

        # fft shift
        try:
            obj = fftshift(obj)
        except ValueError:
            obj = None

    else:
        obj = None

    # Center and crop data
    if auto_center_resize:
        if iobs.ndim is 3:
            nz0, ny0, nx0 = iobs.shape

            # Find center of mass
            z0, y0, x0 = center_of_mass(iobs)
            print("Center of mass at:", z0, y0, x0)
            iz0, iy0, ix0 = int(round(z0)), int(
                round(y0)), int(round(x0))

            # Max symmetrical box around center of mass
            nx = 2 * min(ix0, nx0 - ix0)
            ny = 2 * min(iy0, ny0 - iy0)
            nz = 2 * min(iz0, nz0 - iz0)

            if max_size is not None:
                nx = min(nx, max_size)
                ny = min(ny, max_size)
                nz = min(nz, max_size)

            # Crop data to fulfill FFT size requirements
            nz1, ny1, nx1 = smaller_primes(
                (nz, ny, nx), maxprime=7, required_dividers=(2,))

            print("Centering & reshaping data: (%d, %d, %d) -> \
                (%d, %d, %d)" % (nz0, ny0, nx0, nz1, ny1, nx1))
            iobs = iobs[
                iz0 - nz1 // 2:iz0 + nz1 // 2,
                iy0 - ny1 // 2:iy0 + ny1 // 2,
                ix0 - nx1 // 2:ix0 + nx1 // 2]
            if mask is not None:
                mask = mask[
                    iz0 - nz1 // 2:iz0 + nz1 // 2,
                    iy0 - ny1 // 2:iy0 + ny1 // 2,
                    ix0 - nx1 // 2:ix0 + nx1 // 2]
                print("Centering & reshaping mask: (%d, %d, %d) -> \
                    (%d, %d, %d)" % (nz0, ny0, nx0, nz1, ny1, nx1))

        else:
            ny0, nx0 = iobs.shape

            # Find center of mass
            y0, x0 = center_of_mass(iobs)
            iy0, ix0 = int(round(y0)), int(round(x0))
            print("Center of mass (rounded) at:", iy0, ix0)

            # Max symmetrical box around center of mass
            nx = 2 * min(ix0, nx0 - ix0)
            ny = 2 * min(iy0, ny0 - iy0)
            if max_size is not None:
                nx = min(nx, max_size)
                ny = min(ny, max_size)
                nz = min(nz, max_size)

            # Crop data to fulfill FFT size requirements
            ny1, nx1 = smaller_primes(
                (ny, nx), maxprime=7, required_dividers=(2,))

            print("Centering & reshaping data: (%d, %d) -> (%d, %d)" %
                  (ny0, nx0, ny1, nx1))
            iobs = iobs[iy0 - ny1 // 2:iy0 + ny1 //
                        2, ix0 - nx1 // 2:ix0 + nx1 // 2]

            if mask is not None:
                mask = mask[iy0 - ny1 // 2:iy0 + ny1 //
                            2, ix0 - nx1 // 2:ix0 + nx1 // 2]

    # Create cdi object with data and mask, load the main parameters
    cdi = CDI(
        iobs,
        support=support,
        obj=obj,
        mask=mask,
        wavelength=wavelength,
        pixel_size_detector=pixel_size_detector,
        detector_distance=detector_distance,
    )

    return cdi


def save_cdi_operator_as_cxi(
    gwaihir_dataset,
    cdi_operator,
    path_to_cxi,
):
    """
    We need to create a dictionnary with the parameters to save in the
    cxi file.

    :param cdi_operator: cdi object
     created with PyNX
    :param path_to_cxi: path to future cxi data
     Below are parameters that are saved in the cxi file
        - filename: the file name to save the data to
        - iobs: the observed intensity
        - wavelength: the wavelength of the experiment (in meters)
        - detector_distance: the detector distance (in meters)
        - pixel_size_detector: the pixel size of the detector (in meters)
        - mask: the mask indicating valid (=0) and bad pixels (>0)
        - sample_name: optional, the sample name
        - experiment_id: the string identifying the experiment, e.g.:
          'HC1234: Siemens star calibration tests'
        - instrument: the string identifying the instrument, e.g.:
         'ESRF id10'
        - iobs_is_fft_shifted: if true, input iobs (and mask if any)
        have their origin in (0,0[,0]) and will be shifted back to
        centered-versions before being saved.
        - process_parameters: a dictionary of parameters which will
          be saved as a NXcollection

    :return: Nothing, a CXI file is created.
    """
    cdi_parameters = params
    cdi_parameters["data"] = gwaihir_dataset.iobs
    cdi_parameters["wavelength"] = gwaihir_dataset.wavelength
    cdi_parameters["detector_distance"] = gwaihir_dataset.detector_distance
    cdi_parameters["pixel_size_detector"] = gwaihir_dataset.pixel_size_detector
    cdi_parameters["wavelength"] = gwaihir_dataset.wavelength
    cdi_parameters["verbose"] = gwaihir_dataset.verbose
    cdi_parameters["live_plot"] = gwaihir_dataset.live_plot
    # cdi_parameters["gpu"] = gwaihir_dataset.gpu
    cdi_parameters["auto_center_resize"] = gwaihir_dataset.auto_center_resize
    # cdi_parameters["roi_user"] = gwaihir_dataset.roi_user
    # cdi_parameters["roi_final"] = gwaihir_dataset.roi_final
    cdi_parameters["nb_run"] = gwaihir_dataset.nb_run
    cdi_parameters["max_size"] = gwaihir_dataset.max_size
    # cdi_parameters["data2cxi"] = gwaihir_dataset.data2cxi
    cdi_parameters["output_format"] = "cxi"
    cdi_parameters["mask"] = gwaihir_dataset.mask
    cdi_parameters["support"] = gwaihir_dataset.support
    # cdi_parameters["support_autocorrelation_threshold"]\
    # = gwaihir_dataset.support_autocorrelation_threshold
    cdi_parameters["support_only_shrink"] = gwaihir_dataset.support_only_shrink
    cdi_parameters["object"] = gwaihir_dataset.obj
    cdi_parameters["support_update_period"] = gwaihir_dataset.support_update_period
    cdi_parameters["support_smooth_width_begin"] = gwaihir_dataset.support_smooth_width[0]
    cdi_parameters["support_smooth_width_end"] = gwaihir_dataset.support_smooth_width[1]
    # cdi_parameters["support_smooth_width_relax_n"] = \
    # gwaihir_dataset.support_smooth_width_relax_n
    # cdi_parameters["support_size"] = gwaihir_dataset.support_size
    cdi_parameters["support_threshold"] = gwaihir_dataset.support_threshold
    cdi_parameters["positivity"] = gwaihir_dataset.positivity
    cdi_parameters["beta"] = gwaihir_dataset.beta
    cdi_parameters["crop_output"] = 0
    cdi_parameters["rebin"] = gwaihir_dataset.rebin
    # cdi_parameters["support_update_border_n"] \
    # = gwaihir_dataset.support_update_border_n
    # cdi_parameters["support_threshold_method"] \
    # = gwaihir_dataset.support_threshold_method
    cdi_parameters["support_post_expand"] = gwaihir_dataset.support_post_expand
    cdi_parameters["psf"] = gwaihir_dataset.psf
    # cdi_parameters["note"] = gwaihir_dataset.note
    try:
        cdi_parameters["instrument"] = gwaihir_dataset.beamline
    except AttributeError:
        cdi_parameters["instrument"] = None
    cdi_parameters["sample_name"] = gwaihir_dataset.sample_name
    # cdi_parameters["fig_num"] = gwaihir_dataset.fig_num
    # cdi_parameters["algorithm"] = gwaihir_dataset.algorithm
    cdi_parameters["zero_mask"] = "auto"
    cdi_parameters["nb_run_keep"] = gwaihir_dataset.nb_run_keep
    # cdi_parameters["save"] = gwaihir_dataset.save
    # cdi_parameters["gps_inertia"] = gwaihir_dataset.gps_inertia
    # cdi_parameters["gps_t"] = gwaihir_dataset.gps_t
    # cdi_parameters["gps_s"] = gwaihir_dataset.gps_s
    # cdi_parameters["gps_sigma_f"] = gwaihir_dataset.gps_sigma_f
    # cdi_parameters["gps_sigma_o"] = gwaihir_dataset.gps_sigma_o
    # cdi_parameters["iobs_saturation"] = gwaihir_dataset.iobs_saturation
    # cdi_parameters["free_pixel_mask"] = gwaihir_dataset.free_pixel_mask
    # cdi_parameters["support_formula"] = gwaihir_dataset.support_formula
    # cdi_parameters["mpi"] = "run"
    # cdi_parameters["mask_interp"] = gwaihir_dataset.mask_interp
    # cdi_parameters["confidence_interval_factor_mask_min"] \
    # = gwaihir_dataset.confidence_interval_factor_mask_min
    # cdi_parameters["confidence_interval_factor_mask_max"] \
    # = gwaihir_dataset.confidence_interval_factor_mask_max
    # cdi_parameters["save_plot"] = gwaihir_dataset.save_plot
    # cdi_parameters["support_fraction_min"] \
    # = gwaihir_dataset.support_fraction_min
    # cdi_parameters["support_fraction_max"] \
    # = gwaihir_dataset.support_fraction_max
    # cdi_parameters["support_threshold_auto_tune_factor"] \
    # = gwaihir_dataset.support_threshold_auto_tune_factor
    # cdi_parameters["nb_run_keep_max_obj2_out"] \
    # = gwaihir_dataset.nb_run_keep_max_obj2_out
    # cdi_parameters["flatfield"] = gwaihir_dataset.flatfield
    # cdi_parameters["psf_filter"] = gwaihir_dataset.psf_filter
    cdi_parameters["detwin"] = gwaihir_dataset.detwin
    cdi_parameters["nb_raar"] = gwaihir_dataset.nb_raar
    cdi_parameters["nb_hio"] = gwaihir_dataset.nb_hio
    cdi_parameters["nb_er"] = gwaihir_dataset.nb_er
    cdi_parameters["nb_ml"] = gwaihir_dataset.nb_ml
    try:
        cdi_parameters["specfile"] = gwaihir_dataset.specfile_name
    except AttributeError:
        pass
    # cdi_parameters["imgcounter"] = gwaihir_dataset.imgcounter
    # cdi_parameters["imgname"] = gwaihir_dataset.imgname
    cdi_parameters["scan"] = gwaihir_dataset.scan

    print(
        "\nSaving phase retrieval parameters selected "
        "in the PyNX tab in the cxi file ..."
    )
    cdi_operator.save_data_cxi(
        filename=path_to_cxi,
        process_parameters=cdi_parameters,
    )


def list_reconstructions(
    folder,
    scan_name
):
    """List all cxi files in the folder and sort by creation time"""
    cxi_file_list = [f for f in sorted(
        glob.glob(folder + "*.cxi"),
        key=os.path.getmtime,
        reverse=True,
    ) if not os.path.basename(f).startswith(scan_name)]

    print(
        "################################################"
        "################################################"
    )
    for j, f in enumerate(cxi_file_list):
        file_timestamp = datetime.fromtimestamp(
            os.path.getmtime(f)).strftime('%Y-%m-%d %H:%M:%S')
        print(
            f"File: {os.path.basename(f)}"
            f"\n\tCreated: {file_timestamp}"
        )
        if j != len(cxi_file_list)-1:
            print("")
        else:
            print(
                "################################################"
                "################################################"
            )


def run_modes_decomposition(
    path_scripts,
    folder,
):
    """
    Decomposes several phase retrieval solutions into modes, saves only
    the first mode to save space.

    :param path_scripts: absolute path to script containing
     folder
    :param folder: path to folder in which are stored
     the .cxi files, all files corresponding to
     *LLK* pattern are loaded
    """
    try:
        print(
            "\n###########################################"
            "#############################################"
            f"\nUsing {path_scripts}/pynx-cdi-analysis"
            f"\nUsing {folder}/*LLK* files."
            f"\nRunning: $ pynx-cdi-analysis *LLK* modes=1"
            f"\nOutput in {folder}/modes_gui.h5"
            "\n###########################################"
            "#############################################"
        )
        os.system(
            "{}/pynx-cdi-analysis {}/*LLK* modes=1 modes_output={}/modes_gui.h5".format(
                quote(path_scripts),
                quote(folder),
                quote(folder),
            )
        )
    except KeyboardInterrupt:
        hash_print("Decomposition into modes stopped by user...")
