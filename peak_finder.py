i#copyright 2017, Lars G.


"""A demo implementation of a peak finding algorithm."""


import numpy as np


class PeakFinder:
    """Find and filter peaks inside a vector.

    Parameters
    ----------
    vec : np.ndarray
        Vector to search.
    distance : int
        Required minimal distance between peaks in samples.
    height : float | tuple
        Required height of peaks. Can be one (min) or two values (min, max).
    threshold : float | tuple
        Required height difference between a peak and its adjacent samples. Can
        be one (min) or two values (min, max).
    prominence : float | tuple
        Required prominence of peaks. Can be one (min) or two values (min, max).
        The prominence of a peak is the minimum vertical distance to its
        surrounding valleys.
    width : float | tuple
        Required width of peaks. Can be one (min) or two values (min, max). To
        calculate the width of a peak a horizontal line is drawn at a height
        relative to the peak height (see `wtype`) that intersects with the up-
        and downward slope of the peaks in two points. The horizontal distance
        between these points is the peak width defined by `wtype`.
    wheight : {"halfprom", "halfheight"}
        Two possible width types can be give:

        * "halfprom" - The width is calculated at half the peak prominence.
        * "halfheight" - The width is calculated at half the peak height.

        This argument is ignored if `width` is ``None``.

    Attributes
    ----------
    vec : np.ndarray
        Vector with peaks.
    peaks : np.ndarray
        Positions of filtered peaks in `vec`.
    unfiltered_peaks : np.ndarray
        Positions of unfiltered peaks in `vec`.
    """

    @classmethod
    def get(cls, vec, *args, **kwargs):
        """Find indices of local maxima.

        Parameters
        ----------
        vec : np.ndarray
            Vector to search.
        *args, **kwargs :
            Same as class constructor.

        Returns
        -------
        peaks : np.ndarray
            Positions of peaks in `vec`.
        """
        return cls(vec, *args, **kwargs).peaks

    def __init__(self, vec, height=None, threshold=None, distance=None,
                 prominence=None, width=None, wheight="halfprom"):
        self.vec = vec
        self.peaks, self._valleys = extrema(vec)
        self.unfiltered_peaks = self.peaks

        self._proms = None
        self._width = {"halfprom": None, "halfheight": None}
        self._ax = None

        if height is not None:
            if np.size(height) < 2:
                height = (height, None)
            self.filter_by_height(*height)

        if threshold is not None:
            self.filter_by_threshold(threshold)

        if distance is not None:
            self.filter_by_distance(distance)

        if prominence is not None:
            if np.size(prominence) < 2:
                prominence = (prominence, None)
            self.filter_by_prominence(*prominence)

        if width is not None:
            if np.size(width) < 2:
                width = (width, None)
            self.filter_by_width(*width, wheight=wheight)

    def filter_by_height(self, hmin=None, hmax=None):
        """Remove peaks by there height.

        Parameters
        ----------
        hmin, hmax : float
            Remove peaks whose height is not inside the interval specified by
            these arguments.

        Returns
        -------
        peaks : np.ndarray
            Peaks that weren't removed.
        heights : np.ndarray
            The height of kept `peaks`.
        """
        keep = np.ones(self.peaks.size, dtype=bool)
        if hmin is not None:
            keep &= (self.vec[self.peaks] >= hmin)
        if hmax is not None:
            keep &= (self.vec[self.peaks] <= hmax)
        self._remove_peaks(keep)

        return self.peaks, self.vec[self.peaks]

    def filter_by_threshold(self, tmin):
        """Remove peaks by vertical distance to their neighbouring samples.

        Parameters
        ----------
        tmin : float
            Peaks whose vertical distance to theig neighbouring samples is
            smaller than this are removed..

        Returns
        -------
        peaks : np.ndarray
            Peaks that weren't removed.
        thresholds : np.ndarray
            Minimal vertical distance of each peak to their neighbours.
        """
        thresholds = np.min(np.vstack([
            self.vec[self.peaks] - self.vec[self.peaks - 1],
            self.vec[self.peaks] - self.vec[self.peaks + 1]
        ]), axis=0)
        keep = (thresholds >= tmin)
        self._remove_peaks(keep)

        return self.peaks, thresholds

    def filter_by_distance(self, dmin):
        """Remove peaks by there distance to each other.

        Parameters
        ----------
        dmin : int
            Minimal distance that peaks must be spaced.

        Returns
        -------
        peaks : np.ndarray
            Peaks that weren't removed.
        """
        # Peaks are evaluated by amplitude (larger first)
        eval_peaks = self.peaks[np.argsort(self.vec[self.peaks])][::-1]

        # Flag peaks for deletion
        del_flag = np.zeros(eval_peaks.size, dtype=bool)
        for i in range(eval_peaks.size):
            if not del_flag[i]:
                # Flag peaks in intervall +-distance around current peak
                del_flag |= (eval_peaks >= eval_peaks[i] - dmin) \
                            & (eval_peaks <= eval_peaks[i] + dmin)
                # Keep current peak
                del_flag[i] = False

        keep = ~del_flag[np.argsort(eval_peaks)]
        self._remove_peaks(keep)

        return self.peaks

    def filter_by_prominence(self, pmin=None, pmax=None, wlen=None):
        """Remove peaks by there prominence.

        The prominence of a peak is defined as the vertical distance between the
        peak and ist lowest contour line. The contour line is defined as the
        lowest horizontal line that intersects the rising and falling peak slope
        in two points but contains no higher second peak between these points.

        Parameters
        ----------
        pmin, pmax : float
            Remove peaks whose prominence is not inside the interval specified by
            these arguments.
        wlen : int
            A window lenght in samples that limits the search for the lowest
            contour line two a symmetric interval around the evaluated peak. If
            not given the entire vector is used. Use this parameter to speed up
            the calculation significantly for large vectors.

        Returns
        -------
        peaks : np.ndarray
            Peaks that weren't removed.
        proms : np.ndarray
            The prominences of kept `peaks`.
        """
        self.prominences(wlen)
        keep = np.ones(self.peaks.size, dtype=bool)
        if pmin is not None:
            keep &= (pmin <= self._proms)
        if pmax is not None:
            keep &= (self._proms <= pmax)
        self._remove_peaks(keep)

        return self.peaks, self._proms

    def filter_by_width(self, wmin=None, wmax=None, wheight="halfprom"):
        """Remove peaks by there width.

        Parameters
        ----------
        wmin, wmax : float
            Remove peaks whose width is not inside the interval specified by
            these arguments.
        wheight : {"halfprom", "fullprom"}
            Chooses the height at which the peak width is measured:

            * "halfprom" - The width is measured at half the peak prominence.
            * "fullprom" - The width is measured at the contour line of the
              peak.

        Returns
        -------
        peaks : np.ndarray
            Peaks that weren't removed.
        widths : np.ndarray
            The widths of kept `peaks`.
        """
        widths, *_ = self.widths(wheight)
        keep = np.ones(self.peaks.size, dtype=bool)
        if wmin is not None:
            keep &= (wmin <= widths)
        if wmax is not None:
            keep &= (widths <= wmax)
        self._remove_peaks(keep)

        return self.peaks, widths[keep]

    def prominences(self, wlen=None):
        """Calculate prominences for each peak.

        The prominence of a peak is defined as the vertical distance between the
        peak and ist lowest contour line. The contour line is defined as the
        lowest horizontal line that intersects the rising and falling peak slope
        in two points but contains no higher second peak between these points.

        Parameters
        ----------
        wlen : int
            A window lenght in samples that limits the search for the lowest
            contour line two a symmetric interval around the evaluated peak. If
            not given the entire vector is used. Use this parameter to speed up
            the calculation significantly for large vectors.

        Returns
        -------
        prominences : np.ndarray
            The calculated prominences for each peak (matching
            ``PeakFinder.peaks``).
        """
        if wlen is not None and wlen >= self.vec.size:
            raise ValueError("window lenght must be smaller than vector length")

        # Prepare empty vector
        self._proms = np.zeros(self.peaks.size)
        for i, peak in enumerate(self.peaks):

            if wlen is not None:
                # Calculate window borders around the evaluated peak
                wleft = peak - int(wlen / 2)
                wright = peak + int(wlen / 2)
                # Handle border cases
                if wleft < 0:
                    wright -= wleft
                    wleft = 0
                if self.vec.size < wleft:
                    wleft -= wright - self.vec.size
                    wright = self.vec.size
                vec = self.vec[wleft:wright]
                # Correct peak position in vector
                peak -= wleft
            else:
                # Use full vector for prominence calculation
                vec = self.vec

            # Positions where vector is larger than current peak height
            greater_peak = np.where(vec > vec[peak])[0]

            try:
                # Nearest position to the left of peak with
                # vector[left] > vector[peak]
                left = greater_peak[greater_peak < peak].max()
            except ValueError:
                left = None
            try:
                # Nearest position to right of peak with
                # vector[right] > vector[peak]
                right = greater_peak[greater_peak > peak].min()
            except ValueError:
                right = None

            # # Alternative approach
            # left = zrc.searchsorted(peak) - 1
            # right = zrc.searchsorted(peak, side="right")
            # if left < 0:
            #     left = None
            # else:
            #     left = zrc[left]
            # if right == zrc.size:
            #     right = None
            # else:
            #     right = zrc[right]

            # Contour levels are minima in left and right interval
            left_contour = vec[left:peak].min()
            right_contour = vec[peak:right].min()
            # Select highest contour and calculate vertical distance to peak
            self._proms[i] = vec[peak] - max(left_contour, right_contour)

        return self._proms

    def widths(self, wheight="halfprom"):
        """Calculate peak widths.

        Parameters
        ----------
        wheight : {"halfprom", "fullprom"}
            Chooses the height at which the peak width is measured:

            * "halfprom" - The width is measured at half the peak prominence.
            * "fullprom" - The width is measured at the contour line of the
              peak.

        Returns
        -------
        widths : np.ndarray
            The widths for each peak.
        width_heights : np.ndarray
            The heights at which the `widths` where measured.
        left, right : np.ndarray
            Positions of left and right intersection points of a horizontal line
            at `width_heights`.
        """
        if wheight == "halfprom":
            if self._proms is None:
                self.prominences()
            width_height = self.vec[self.peaks] - self._proms / 2
        elif wheight == "fullprom":
            if self._proms is None:
                self.prominences()
            width_height = self.vec[self.peaks] - self._proms
        else:
            raise ValueError("value of argument wheight is unsupported")

        widths = np.zeros(self.peaks.size)
        left_pos = np.zeros(widths.size, dtype=int)
        right_pos = np.zeros(widths.size, dtype=int)

        for i, (peak, height) in enumerate(zip(self.peaks, width_height)):

            # Positions where vector is smaller reference height
            is_smaller = np.where(self.vec <= height)[0]

            try:
                # Nearest position to the left of peak with
                # vector[left] > vector[peak]
                left = is_smaller[is_smaller < peak].max()
            except ValueError:
                left = 0
            try:
                # Nearest position to right of peak with
                # vector[right] > vector[peak]
                right = is_smaller[is_smaller > peak].min()
            except ValueError:
                right = self.vec.size

            widths[i] = right - left
            left_pos[i] = left
            right_pos[i] = right

        return widths, width_height, left_pos, right_pos

    def reset(self):
        """Remove filters and restore all peaks."""
        self.peaks = self.unfiltered_peaks
        self._proms = None

    def plot(self, ax=None):
        """Plot results of peak detection."""
        import matplotlib.pyplot as plt

        if ax is not None:
            pass
        elif self._ax:
            ax.clear()
        else:
            _, ax = plt.subplots()

        ax.plot(self.vec)
        ax.plot(self.peaks, self.vec[self.peaks], "x")
        ax.set_xlabel("Sample number")
        ax.set_ylabel("Amplitude")
        ax.legend([ax.lines[1]], [f"{self.peaks.size} peaks"])

        if self._proms is not None:
            ymax = self.vec[self.peaks]
            ymin = ymax - self._proms
            ax.vlines(self.peaks, ymin, ymax, colors="C1")

        return ax

    def _remove_peaks(self, keep):
        """Only keep peaks in `keep`."""
        self.peaks = self.peaks[keep]
        if self._proms is not None:
            self._proms = self._proms[keep]


def extrema(vec):
    """Find indices of all local maxima and minima in vector.

    Parameters
    ----------
    vec : np.array, one-dimensional
        Vector to process.

    Returns
    -------
    minima : np.ndarray
        All local maxima in `vec`.
    maxima : np.ndarray
        All local minima in `vec`.

    See Also
    --------
    PeakFinder
    """
    vec = validate_vector(vec)

    # Calculate trends of 1. differences in vector
    sdiff = np.sign(np.diff(vec))
    # Symmetrically close zero gaps (plateaus)
    sdiff = _close_zero_gap(sdiff)

    rising = sdiff == 1
    falling = sdiff == -1

    # Find extrema with patterns [.. 1 -1 ..] and [.. -1 1 ..]
    maxima = np.where(rising[:-1] & falling[1:])[0] + 1
    minima = np.where(falling[:-1] & rising[1:])[0] + 1

    return maxima, minima


def _close_zero_gap(vec, copy=False):
    """Symetrically close zero gaps.

    Closes zero gaps by propagating the edge values.

    Parameters
    ----------
    vec : np.ndarray[int]
        Vector with values -1, 0 or 1.

    Returns
    -------
    vec : np.ndarray
        Vector without zero gaps.

    Examples
    --------

    >>> vec = np.array([1, 0, 0, 0, -1, 1, 0, 0, 1, -1,  0, 0, 0, 0, -1, 0, 1])
    >>> vec, _close_zero_gap(vec)
    (array([ 1,  0,  0,  0, -1,  1,  0,  0,  1, -1,  0,  0,  0,  0, -1,  0,  1]),
     array([ 1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1, -1, -1, -1, -1, -1,  1]))
    """
    if not np.any(vec):
        raise ValueError("can't close gap in vector with all zeros")
    if copy is True:
        vec = vec.copy()

    while True:
        const = vec == 0
        # Break if no zeros remain
        if not np.any(const):
            break
        rising = vec == 1
        falling = vec == -1

        # Remove zeros on both edges
        vec[:-1][const[:-1] & rising[1:]] = 1    # [.. 0 1 ..] -> [.. 1 1 ..]
        vec[:-1][const[:-1] & falling[1:]] = -1  # [.. 0 -1 ..] -> [.. -1 -1 ..]
        vec[1:][rising[:-1] & const[1:]] = 1     # [.. 1 0 ..] -> [.. 1 1 ..]
        vec[1:][falling[:-1] & const[1:]] = -1   # [.. -1 0 ..] -> [.. -1 -1 ..]

    return vec


def validate_vector(x):
    """Ensure that `x` is a vector like object.

    Parameters
    ----------
    x : array_like or iterable
        Object to be validated.

    Returns
    -------
    x : np.ndarray
        Returns `x` if already a np.ndarray otherwise the array version of `x` is
        returned.

    Raises
    ------
    ValuError
        If `x` is not a one-dimensional array or iterable object.

    See Also
    --------
    validate_vectors
    """
    x = np.asarray(x)
    if not x.ndim == 1:
        raise ValueError(f"not a vector, dimension was {x.ndim}")
    return x

