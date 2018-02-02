# -*- coding: utf-8 -*-
'''
The following code was authored by Jiliang Liu (jiliangliu@bnl.gov), and
imported into this library for convenience.

Some changes:

'''
import numpy as np
import statsmodels.api as sm
import sys
from lmfit.models import GaussianModel
# form lmfit.models import ExponentialModel
# form lmfit.models import LorentzianModel
# form lmfit.models import PolynomialModel
from scipy.interpolate import interp1d


''' Code review:
        Overall, it is not easy to read. I think making it more readable would
        increase the number of people using it. Here are some suggestions and
        notes on what I've changed.

    Suggestions:
    do not use beam_stop_effect and end
        the user could specify an array that is already sliced
        before hand
    rename beam_stop_effect to "start"
    I would rename the output variables to things that might make more sense
    Add the description for the output variables in the "Returns" section

    Notes:
        - removed bkdg_span because it did not seem to be used in this version
        - removed plotting references
        - removed from pylab import *
            (so use np.nan, np.size, not nan and size)
        - renamed some variables to shorten lines
        - removed the kwargs and explictly wrote out kwargs
            it is then easier to read. also, if user specifies
            a kwarg that is not accepted it will raise an error
'''


class peak_finding:
    ''' Peak finding

        This class will find peaks in 1D intensity data

        Parameters
        ----------
        intensity : 1d np.ndarray
            the intensity to find the peaks
        limit_fev : bool, optional
            decided whether or not to limit function evaluations
            in the optimization
        beam_stop_effect : int, optional
            the first few pixels to ignore (from the beam stop
            effect).
        end: int, optional
            the last point for the intensity curve to look at
        adjust_coefficient : float, optional
            adjust this coefficient to look for sharper peaks (larger)
        frac : float, optional
            This is the input to the Locally Weighted Scatterplot Smoothing on
            the data.
            Between 0 and 1. The graction of the data used when estimating each
            y-value
        it : int
            This is the input to the Locally Weighted Scatterplot Smoothing on
            the data.
            The number of residual-based reweightings to perform.
        delta : float
            T
            Distance within which to use linear-interpolation instead of
            weighted regression.
        frac_over_smooth : float, optional
            This is the input to the Locally Weighted Scatterplot Smoothing on
            the smoothed data.
            Between 0 and 1. The graction of the data used when estimating each
            y-value
        it_over_smooth :
            This is the input to the Locally Weighted Scatterplot Smoothing on
            the smoothed data.
            The number of residual-based reweightings to perform.
        max_peaks : int, optional
            The maximum number of peaks to try to fit to

        Returns
        -------
        return out :
        y_origin[beam_stop_effect:end] :
        inds_peak
        xdata
        ratio
        z[:, 1]
        y
        w
        bkgd
        variance
        variance_mean

        Examples
        --------
        y = np.random.random((100,))
        >>> result = peak_finding(intensity=y).peak_position()
        >>> peak_positions = result[1]

    '''
    def __init__(self, intensity=None, limit_fev=False, beam_stop_effect=0,
                 end=None, adjust_coefficient=2, frac=0.06/3, it=5, delta=0,
                 frac_over_smooth=1.8/3, it_over_smooth=5, max_peaks=10):
        # this was not used so I comment it out
        # , bkgd_span=20):
        self.limit_fev = limit_fev
        if intensity is None:
            print('not intensity input')
            sys, exit()
        else:
            self.intensity = intensity

        # set beam stop threshold
        self.beam_stop_effect = beam_stop_effect
        # set end of intensity array
        if end is None:
            end = len(self.intensity)
        self.end = end
        # set adjust_coefficient which determine the significancy of
        # distinguable reflection. large adjust_coefficient only allow the
        # algorithm to identify very sharp peaks
        self.adjust_coefficient = adjust_coefficient
        # set fraction of the data for smooth
        self.frac = frac
        # set iteration for smooth
        self.it = it
        # set delta for local regression fitting, delta may introduced if
        # furhter smooth or iteration needed.
        self.delta = delta
        # set fraction of the data for over smooth, over smooth was enable an
        # approximate setimation of background
        self.frac_over_smooth = frac_over_smooth
        # set iteration for over smooth
        self.it_over_smooth = it_over_smooth
        self.max_peaks = max_peaks
        # set the span used for background fitting, the span too small may
        # introduce noise, but too large may lead to not efficient background
        # estimation
        # this is not used to I comment it out
        # self.bkdg_span = bkgd_span

    def peak_position(self, optim_bkgd=True, iter_num=2):
        # assign value to variants
        intensity = np.copy(self.intensity)
        beam_stop_effect = self.beam_stop_effect
        frac = self.frac
        it = self.it
        delta = self.delta
        frac_over_smooth = self.frac_over_smooth
        it_over_smooth = self.it_over_smooth
        end = self.end
        # adjust_coefficient = self.adjust_coefficient

        x_origin = np.arange(0, len(intensity), 1)
        if beam_stop_effect > 0:
            intensity[0:beam_stop_effect] = np.nan
        else:
            pass
        intensity[intensity < -6] = np.nan
        y_origin = np.copy(intensity)
        not_nan_inds = np.where(np.isnan(y_origin) == 0)[0]
        # return not_nan_inds
        if np.size(not_nan_inds) == 0:
            return []
        else:
            pass
        if not_nan_inds[0] > beam_stop_effect:
            beam_stop_effect = not_nan_inds[0]
        if not_nan_inds[-1] < end:
            end = not_nan_inds[-1]
        # return end,x_origin,y_origin,beam_stop_effect
        # linear interpolate the missing point in middle of intensity curve
        nan_ys = np.where(np.isnan(y_origin[beam_stop_effect:end]))[0]
        if np.size(nan_ys) > 0:
            # this time it's up to end+1? redefine
            clip = slice(beam_stop_effect, end+1)
            y_origin_clipped = y_origin[clip]
            x_origin_clipped = x_origin[clip]
            nan_ys, = np.where(np.isnan(y_origin_clipped))
            fb = interp1d(x_origin_clipped[nan_ys], y_origin_clipped[nan_ys],
                          kind='slinear')
            y_origin_clipped = fb(x_origin_clipped)
        else:
            pass
        # nan should only appear at origin and end
        # not_nan_inds = np.arange(beam_stop_effect,end+1,1)
        # get the good ys
        not_nan_inds, = np.where(~np.isnan(y_origin))
        # y_origin had been interpolated, thus need new not_nan_ind
        x = x_origin[not_nan_inds]
        y = y_origin[not_nan_inds]
        # locally weighted scatterplot smoothing
        z = sm.nonparametric.lowess(y, x, frac=frac_over_smooth,
                                    it=it_over_smooth)
        w = sm.nonparametric.lowess(y, x, frac=frac, it=it, delta=delta)
        # return w,intensity,y_origin,x_origin
        w_grad = np.gradient(w[:, 1], 3)
        w_grad_sign = np.sign(w_grad)
        turn = w_grad_sign[0:-2] - w_grad_sign[1:-1]
        # turn = \
        #   np.sign(np.gradient(y))[0:end-1]-np.sign(np.gradient(y))[1:end]
        local_minimum = np.where(turn < 0)[0]
        normalized_gradient = np.gradient(w[:, 1]/sum(w[:, 1]))

        if len(normalized_gradient) < 21.:
            b_c_width = len(normalized_gradient)
        else:
            b_c_width = 21.

        # applied low pass filt, which enable to find the mean value of local
        # region with width of 21 pixel and center at corresponding place
        # SG smoothing maybe?
        conv_filt = np.ones((int(b_c_width), ))/b_c_width
        low_pass_filt_gradient = np.convolve(normalized_gradient,
                                             conv_filt, 'same')
        # standard deviation of local region is calculated by invovling a
        # convolvution of boxfunction with width of 21 pixel
        # rename this to something more appropriate
        _ = np.abs(normalized_gradient - low_pass_filt_gradient)
        gradient_local_std = np.convolve(_, conv_filt, 'same')
        # eliminate the incorrect local minimum
        # local_minimum =
        # local_minimum[np.where(gradient_local_std[local_minimum]<np.sum(gradient_local_std)/len(gradient_local_std))[0]]
        #
        flat_region,  = \
            np.where(gradient_local_std < .005*np.average(gradient_local_std))
        # 1e-6 is applied to determine the flat rate of data, it is a empirical
        # parameter here
        local_minimum = np.unique(np.append(local_minimum, flat_region))
        if np.size(local_minimum) == 0:
            return []
        else:
            pass
        turn_minimum_exist = False
        if local_minimum[0] > 0:
            if w[local_minimum[0], 1] < w[0, 1]:
                local_minimum = np.append(np.arange(0, local_minimum[0], 1),
                                          local_minimum)
            else:
                turn_minimum_exist = True
                turns = np.diff(np.sign(np.gradient(gradient_local_std)))
                turn_minimum, = np.where(turns > 0)
                # actually this may cause an additional unexpect peak between
                # first peak and beam stop
                local_minimum = np.append(np.arange(0, turn_minimum[0]),
                                          local_minimum)
        # return turn_minimum_exist,turn_minimum
        # if local_minimum[0]>beam_stop_effect:
        #    local_minimum = np.append(beam_stop_effect,local_minimum)

        # find local minimum and cubic spline the local minimum as estimated
        # background
        local_minimum = w[local_minimum, 0].astype(int)
        if local_minimum[-1] < w[-1, 0]:
            local_minimum = np.append(local_minimum,
                                      np.arange(local_minimum[-1]+1,
                                                int(w[-1, 0])+1, 1))

        # local_minimum = local_minimum+beam_stop_effect
        bkgd = np.zeros((len(y_origin), ))*np.nan

        y_origin[w[:, 0].astype(int)] = w[:, 1]
        # return y,y_origin,local_minimum,w[:,1],intensity,not_nan_inds,w[:,0]
        f2 = interp1d(local_minimum, y_origin[local_minimum], kind='slinear')
        bkgd[local_minimum[0]:local_minimum[-1]] =\
            f2(np.arange(local_minimum[0], local_minimum[-1], 1))
        bkgd[np.where(y_origin-bkgd < 0)[0]] =\
            y_origin[np.where(y_origin-bkgd < 0)[0]]

        # return w,y_origin,bkgd,x_origin,local_minimum

        optim_bkgd = optim_bkgd
        iter_num = 2
        if len(normalized_gradient) < 81.:
            b_c_width = len(normalized_gradient)
        else:
            b_c_width = 81.
        # b_c_width is background convolve window width, if window is too
        # narrow, will not be able distinguish sharp (which is wanted to
        # eliminated) and diffuse background
        if optim_bkgd is True:
            bkgd_gradient = np.gradient(bkgd[beam_stop_effect:end]
                                        / np.sum(bkgd[beam_stop_effect:end]),
                                        3)
            conv_filt = np.ones((int(b_c_width),))/b_c_width
            local_ave_gradient = np.convolve(bkgd_gradient, conv_filt, 'same')
            local_std_gradeint = \
                np.convolve(np.abs(bkgd_gradient - local_ave_gradient),
                            conv_filt, 'same')
            bkgd_turn = np.sign(bkgd_gradient)[0:-2]\
                - np.sign(bkgd_gradient)[1:-1]
            bkgd_local_minimum = np.where(bkgd_turn < 0)[0] + beam_stop_effect
            if turn_minimum_exist is True:
                bkgd_local_minimum = np.append(turn_minimum[0]
                                               + beam_stop_effect,
                                               bkgd_local_minimum)
            for g in range(iter_num):
                bkgd_gradient = \
                    np.gradient(bkgd[beam_stop_effect:end]
                                / np.sum(bkgd[beam_stop_effect:end]), 3)
                local_ave_gradient = np.convolve(bkgd_gradient,
                                                 conv_filt, 'same')
                local_std_gradeint = \
                    np.convolve(np.abs(bkgd_gradient-local_ave_gradient),
                                conv_filt, 'same')
                # return local_std_gradeint,local_ave_gradient
                if np.size(bkgd_local_minimum) > 0:
                    # return bkgd_local_minimum
                    sel = slice(bkgd_local_minimum[0], -1)
                    bkgd_judge = \
                        np.sum(local_std_gradeint[sel]) / \
                        len(local_std_gradeint[sel])
                    for mini_bkgd in range(len(bkgd_local_minimum)-1):
                        sel = slice(bkgd_local_minimum[mini_bkgd],
                                    bkgd_local_minimum[mini_bkgd + 1])
                        local_mini_std = \
                            np.sum(local_std_gradeint[sel])\
                            / len(local_std_gradeint[sel])
                        if local_mini_std > .6*bkgd_judge:
                            # interpolate between inds
                            x0, x1 = bkgd_local_minimum[mini_bkgd], \
                                bkgd_local_minimum[mini_bkgd+1]
                            xtmp = np.asarray([x0, x1])
                            ytmp = np.asarray([bkgd[x0], bkgd[x1]])
                            f1 = interp1d(xtmp, ytmp, kind='slinear')
                            bkgd[x0:x1] = f1(np.arange(x0, x1))
                    bkgd[np.where(y_origin-bkgd < 0)[0]] = \
                        y_origin[np.where(y_origin-bkgd < 0)[0]]
                else:
                    pass
        variance = y_origin[beam_stop_effect:end]-bkgd[beam_stop_effect:end]
        y_origin_bgsub = y_origin[beam_stop_effect:end] \
            - bkgd[beam_stop_effect:end]
        variance_mean = np.sum(abs(y_origin_bgsub))\
            / len(x_origin[beam_stop_effect:end])
        # here to determine whether the peak is too weak to consider as a
        # reflection from sample, should have intensity large than 2
        # return w,y_origin,bkgd,x_origin
        intensity_threshold = -1
        if np.size(np.where(variance > intensity_threshold)[0]) == 0:
            return []
        else:
            pass

        turn_v = np.sign(np.gradient(variance, 3))[0:-2]\
            - np.sign(np.gradient(variance, 3))[1:-1]
        inds = np.asarray(np.where(turn_v > 0))
        # return inds
        # return variance,variance_mean,inds
        inds_peak = list()
        for num1 in range(len(inds[0, :])):
            if variance[inds[0, num1]] > self.adjust_coefficient*variance_mean:
                inds_peak.append(inds[0, num1])
                # print inds[0,num1]
            elif variance[inds[0, num1] + 1] > \
                    self.adjust_coefficient*variance_mean:
                inds_peak.append(inds[0, num1] + 1)
            # print inds[0,num1]+1
            # else: inds_peak = inds_peak
        # print x[np.unique(np.asarray(inds_peak))+beam_stop_effect]
        # x=np.copy(x_origin[beam_stop_effect:end])
        # x=x[isnan(variance)==0]
        # '''
        # here x should relate to variance not only intensity not nan, there is
        # interpolation for variance
        # '''
        if np.size(inds_peak) == 0:
            return inds_peak
        else:
            inds_peak_amp = inds_peak
            inds_peak = x[np.unique(np.asarray(inds_peak))]
            # return inds_peak,variance,variance_mean,y,bkgd,turn_minimum_exist
        # np.round assume distance between two neighbor peaks need to be larger
        # 1 pixel

        xdata = x_origin[beam_stop_effect:end]

        if len(inds_peak) > self.max_peaks:
            print("Maximum number of peaks exceed. Choosing")
            print(" first {} peaks".format(self.max_peaks))
            inds_peak = inds_peak[:self.max_peaks]

        for num11 in range(len(inds_peak)):
            # print num11
            pre_name = "g%d_" % (num11+1)
            pre_name_center = "g%d_center" % (num11+1)
            pre_name_sigma = "g%d_sigma" % (num11+1)
            pre_name_amplitude = "g%d_amplitude" % (num11+1)
            # pre_name_height = "g%d_height" % (num11+1)
            if num11 == 0:
                mod = GaussianModel(prefix=pre_name)
                pars = GaussianModel(prefix=pre_name).make_params()
            else:
                pars.update(GaussianModel(prefix=pre_name).make_params())
                mod += GaussianModel(prefix=pre_name)
            # limit the peak position shift to ensure the fitting accuracy
            peakguess = inds_peak[num11]
            minpos = inds_peak[num11]-2
            maxpos = inds_peak[num11]+2
            pars[pre_name_center].set(peakguess, min=minpos, max=maxpos)
            # very native and intuitive guess
            pars[pre_name_sigma].set(.5, min=0.001)
            # amplitude guess from intensity of inds_peak
            pars[pre_name_amplitude].set(variance[inds_peak_amp[num11]], min=0)
        # print("Now fitting")
        # print("model : {}".format(mod))
        # raise ValueError
        # 10,000 func calls is max
        # JULIEN ADDED THIS: put a cap on the number of function evaluations.
        # This part seems to take forever if the number of peaks is large
        # Also, if num peaks is large, maybe this is a mistake?
        # Or alternatively, if the number of peaks is large, perhaps we should
        # try to see which ones we might want to pick that seem more likely to
        # be peaks than others by some criterion?
        if self.limit_fev:
            maxevals = 1000
            # number of parameters
            N = len(mod.param_names)
            maxfev = np.minimum(maxevals, 200*(N+1))
            fit_kws = dict(maxfev=maxfev)
        else:
            fit_kws = dict()

        from ..core.timeout import timeout
        try:
            out = timeout(seconds=2)(mod.fit)(variance, pars, x=xdata,
                                              method='leastsq',
                                              fit_kws=fit_kws)
        except Exception:
            out = None

        # return inds_peak,out,turn,x,xdata,y
        wvar, = np.where(variance > variance_mean)
        ratio = np.sum(variance[wvar])/np.sum(bkgd[beam_stop_effect:end][wvar])
        return out, \
            y_origin[beam_stop_effect:end], \
            inds_peak, \
            xdata, \
            ratio, \
            z[:, 1], \
            y, \
            w, \
            bkgd, \
            variance, \
            variance_mean
        # ,gradient_local_std,low_pass_filt_gradient
