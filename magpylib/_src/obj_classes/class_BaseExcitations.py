"""BaseHomMag class code"""
# pylint: disable=cyclic-import
import warnings

import numpy as np

from magpylib import ureg
from magpylib._src.exceptions import MagpylibDeprecationWarning
from magpylib._src.fields.field_wrap_BH import getBH_level2
from magpylib._src.input_checks import check_format_input_scalar
from magpylib._src.input_checks import check_format_input_vector
from magpylib._src.input_checks import validate_field_func
from magpylib._src.obj_classes.class_BaseDisplayRepr import BaseDisplayRepr
from magpylib._src.obj_classes.class_BaseGeo import BaseGeo
from magpylib._src.style import CurrentStyle
from magpylib._src.style import MagnetStyle
from magpylib._src.utility import format_star_input


class BaseSource(BaseGeo, BaseDisplayRepr):
    """Base class for all types of sources. Provides getB and getH methods for source objects
    and corresponding field function"""

    _field_func = None
    _field_func_kwargs = {}
    _editable_field_func = False

    def __init__(self, position, orientation, field_func=None, style=None, **kwargs):
        if field_func is not None:
            self.field_func = field_func
        BaseGeo.__init__(self, position, orientation, style=style, **kwargs)
        BaseDisplayRepr.__init__(self)

    @property
    def field_func(self):
        """
        The function for B- and H-field computation must have the two positional arguments
        `field` and `observers`. With `field='B'` or `field='H'` the B- or H-field in units
        of T or A/m must be returned respectively. The `observers` argument must
        accept numpy ndarray inputs of shape (n,3), in which case the returned fields must
        be numpy ndarrays of shape (n,3) themselves.
        """
        return self._field_func

    @field_func.setter
    def field_func(self, val):
        if self._editable_field_func:
            validate_field_func(val)
        else:
            raise AttributeError(
                "The `field_func` attribute should not be edited for original Magpylib sources."
            )
        self._field_func = val

    def getB(
        self, *observers, squeeze=True, pixel_agg=None, output="ndarray", in_out="auto"
    ):
        """Compute the B-field at observers in units of T generated by the source.

        SI units are used for all inputs and outputs.

        Parameters
        ----------
        observers: array_like or (list of) `Sensor` objects
            Can be array_like positions of shape (n1, n2, ..., 3) where the field
            should be evaluated, a `Sensor` object with pixel shape (n1, n2, ..., 3) or a list
            of such sensor objects (must all have similar pixel shapes). All positions are given
            in units of m.

        squeeze: bool, default=`True`
            If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g.
            only a single source) are eliminated.

        pixel_agg: str, default=`None`
            Reference to a compatible numpy aggregator function like `'min'` or `'mean'`,
            which is applied to observer output values, e.g. mean of all sensor pixel outputs.
            With this option, observers input with different (pixel) shapes is allowed.

        output: str, default='ndarray'
            Output type, which must be one of `('ndarray', 'dataframe')`. By default a multi-
            dimensional array ('ndarray') is returned. If 'dataframe' is chosen, the function
            returns a 2D-table as a `pandas.DataFrame` object (the Pandas library must be
            installed).

        in_out: {'auto', 'inside', 'outside'}
            This parameter only applies for magnet bodies. It specifies the location of the
            observers relative to the magnet body, affecting the calculation of the magnetic field.
            The options are:
            - 'auto': The location (inside or outside the cuboid) is determined automatically for
            each observer.
            - 'inside': All observers are considered to be inside the cuboid; use this for
              performance optimization if applicable.
            - 'outside': All observers are considered to be outside the cuboid; use this for
              performance optimization if applicable.
            Choosing 'auto' is fail-safe but may be computationally intensive if the mix of observer
            locations is unknown.

        Returns
        -------
        B-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3) or DataFrame
            B-field at each path position (index m) for each sensor (index k) and each sensor
            pixel position (indices n1,n2,...) in units of T. Sensor pixel positions are equivalent
            to simple observer positions. Paths of objects that are shorter than index m will be
            considered as static beyond their end.

        Examples
        --------
        Compute the B-field of a spherical magnet at three positions:

        >>> import magpylib as magpy
        >>> src = magpy.magnet.Sphere(polarization=(0,0,1.), diameter=1)
        >>> B = src.getB(((0,0,0), (1,0,0), (2,0,0)))
        >>> print(B)
        [[ 0.          0.          0.66666667]
         [ 0.          0.         -0.04166667]
         [ 0.          0.         -0.00520833]]

        Compute the B-field at two sensors, each one with two pixels

        >>> sens1 = magpy.Sensor(position=(1,0,0), pixel=((0,0,.1), (0,0,-.1)))
        >>> sens2 = sens1.copy(position=(2,0,0))
        >>> B = src.getB(sens1, sens2)
        >>> print(B)
        [[[ 0.01219289  0.         -0.0398301 ]
          [-0.01219289  0.         -0.0398301 ]]
        <BLANKLINE>
         [[ 0.00077639  0.         -0.00515004]
          [-0.00077639  0.         -0.00515004]]]
        """
        observers = format_star_input(observers)
        return getBH_level2(
            self,
            observers,
            field="B",
            sumup=False,
            squeeze=squeeze,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
        )

    def getH(
        self, *observers, squeeze=True, pixel_agg=None, output="ndarray", in_out="auto"
    ):
        """Compute the H-field in units of A/m at observers generated by the source.

        Parameters
        ----------
        observers: array_like or (list of) `Sensor` objects
            Can be array_like positions of shape (n1, n2, ..., 3) where the field
            should be evaluated, a `Sensor` object with pixel shape (n1, n2, ..., 3) or a list
            of such sensor objects (must all have similar pixel shapes). All positions
            are given in units of m.

        squeeze: bool, default=`True`
            If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g.
            only a single source) are eliminated.

        pixel_agg: str, default=`None`
            Reference to a compatible numpy aggregator function like `'min'` or `'mean'`,
            which is applied to observer output values, e.g. mean of all sensor pixel outputs.
            With this option, observers input with different (pixel) shapes is allowed.

        output: str, default='ndarray'
            Output type, which must be one of `('ndarray', 'dataframe')`. By default a multi-
            dimensional array ('ndarray') is returned. If 'dataframe' is chosen, the function
            returns a 2D-table as a `pandas.DataFrame` object (the Pandas library must be
            installed).

        in_out: {'auto', 'inside', 'outside'}
            This parameter only applies for magnet bodies. It specifies the location of the
            observers relative to the magnet body, affecting the calculation of the magnetic field.
            The options are:
            - 'auto': The location (inside or outside the cuboid) is determined automatically for
            each observer.
            - 'inside': All observers are considered to be inside the cuboid; use this for
              performance optimization if applicable.
            - 'outside': All observers are considered to be outside the cuboid; use this for
              performance optimization if applicable.
            Choosing 'auto' is fail-safe but may be computationally intensive if the mix of observer
            locations is unknown.

        Returns
        -------
        H-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3) or DataFrame
            H-field at each path position (index m) for each sensor (index k) and each sensor
            pixel position (indices n1,n2,...) in units of A/m. Sensor pixel positions are
            equivalent to simple observer positions. Paths of objects that are shorter than
            index m will be considered as static beyond their end.

        Examples
        --------
        Compute the H-field of a spherical magnet at three positions:

        >>> import magpylib as magpy

        >>> src = magpy.magnet.Sphere(polarization=(0,0,1.), diameter=1)
        >>> H = src.getH(((0,0,0), (1,0,0), (2,0,0)))
        >>> print(H)
        [[      0.               0.         -265258.23848649]
         [      0.               0.          -33157.27981081]
         [      0.               0.           -4144.65997635]]

        Compute the H-field at two sensors, each one with two pixels

        >>> sens1 = magpy.Sensor(position=(1,0,0), pixel=((0,0,.1), (0,0,-.1)))
        >>> sens2 = sens1.copy(position=(2,0,0))
        >>> H = src.getH(sens1, sens2)
        >>> print(H)
        [[[  9702.7918453       0.         -31695.78669464]
          [ -9702.7918453       0.         -31695.78669464]]
        <BLANKLINE>
         [[   617.83031378      0.          -4098.27441472]
          [  -617.83031378      0.          -4098.27441472]]]

        """
        observers = format_star_input(observers)
        return getBH_level2(
            self,
            observers,
            field="H",
            sumup=False,
            squeeze=squeeze,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
        )

    def getM(
        self, *observers, squeeze=True, pixel_agg=None, output="ndarray", in_out="auto"
    ):
        """Compute the M-field in units of A/m at observers generated by the source.

        Parameters
        ----------
        observers: array_like or (list of) `Sensor` objects
            Can be array_like positions of shape (n1, n2, ..., 3) where the field
            should be evaluated, a `Sensor` object with pixel shape (n1, n2, ..., 3) or a list
            of such sensor objects (must all have similar pixel shapes). All positions
            are given in units of m.

        squeeze: bool, default=`True`
            If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g.
            only a single source) are eliminated.

        pixel_agg: str, default=`None`
            Reference to a compatible numpy aggregator function like `'min'` or `'mean'`,
            which is applied to observer output values, e.g. mean of all sensor pixel outputs.
            With this option, observers input with different (pixel) shapes is allowed.

        output: str, default='ndarray'
            Output type, which must be one of `('ndarray', 'dataframe')`. By default a multi-
            dimensional array ('ndarray') is returned. If 'dataframe' is chosen, the function
            returns a 2D-table as a `pandas.DataFrame` object (the Pandas library must be
            installed).

        in_out: {'auto', 'inside', 'outside'}
            This parameter only applies for magnet bodies. It specifies the location of the
            observers relative to the magnet body, affecting the calculation of the magnetic field.
            The options are:
            - 'auto': The location (inside or outside the cuboid) is determined automatically for
            each observer.
            - 'inside': All observers are considered to be inside the cuboid; use this for
              performance optimization if applicable.
            - 'outside': All observers are considered to be outside the cuboid; use this for
              performance optimization if applicable.
            Choosing 'auto' is fail-safe but may be computationally intensive if the mix of observer
            locations is unknown.

        Returns
        -------
        M-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3) or DataFrame
            M-field at each path position (index m) for each sensor (index k) and each sensor
            pixel position (indices n1,n2,...) in units of A/m. Sensor pixel positions are
            equivalent to simple observer positions. Paths of objects that are shorter than
            index m will be considered as static beyond their end.
        """
        observers = format_star_input(observers)
        return getBH_level2(
            self,
            observers,
            field="M",
            sumup=False,
            squeeze=squeeze,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
        )

    def getJ(
        self, *observers, squeeze=True, pixel_agg=None, output="ndarray", in_out="auto"
    ):
        """Compute the J-field at observers in units of T generated by the source.

        SI units are used for all inputs and outputs.

        Parameters
        ----------
        observers: array_like or (list of) `Sensor` objects
            Can be array_like positions of shape (n1, n2, ..., 3) where the field
            should be evaluated, a `Sensor` object with pixel shape (n1, n2, ..., 3) or a list
            of such sensor objects (must all have similar pixel shapes). All positions are given
            in units of m.

        squeeze: bool, default=`True`
            If `True`, the output is squeezed, i.e. all axes of length 1 in the output (e.g.
            only a single source) are eliminated.

        pixel_agg: str, default=`None`
            Reference to a compatible numpy aggregator function like `'min'` or `'mean'`,
            which is applied to observer output values, e.g. mean of all sensor pixel outputs.
            With this option, observers input with different (pixel) shapes is allowed.

        output: str, default='ndarray'
            Output type, which must be one of `('ndarray', 'dataframe')`. By default a multi-
            dimensional array ('ndarray') is returned. If 'dataframe' is chosen, the function
            returns a 2D-table as a `pandas.DataFrame` object (the Pandas library must be
            installed).

        in_out: {'auto', 'inside', 'outside'}
            This parameter only applies for magnet bodies. It specifies the location of the
            observers relative to the magnet body, affecting the calculation of the magnetic field.
            The options are:
            - 'auto': The location (inside or outside the cuboid) is determined automatically for
            each observer.
            - 'inside': All observers are considered to be inside the cuboid; use this for
              performance optimization if applicable.
            - 'outside': All observers are considered to be outside the cuboid; use this for
              performance optimization if applicable.
            Choosing 'auto' is fail-safe but may be computationally intensive if the mix of observer
            locations is unknown.

        Returns
        -------
        J-field: ndarray, shape squeeze(m, k, n1, n2, ..., 3) or DataFrame
            J-field at each path position (index m) for each sensor (index k) and each sensor
            pixel position (indices n1,n2,...) in units of T. Sensor pixel positions are equivalent
            to simple observer positions. Paths of objects that are shorter than index m will be
            considered as static beyond their end.
        """
        observers = format_star_input(observers)
        return getBH_level2(
            self,
            observers,
            field="J",
            sumup=False,
            squeeze=squeeze,
            pixel_agg=pixel_agg,
            output=output,
            in_out=in_out,
        )


class BaseMagnet(BaseSource):
    """provides the magnetization and polarization attributes for magnet classes"""

    _style_class = MagnetStyle

    def __init__(
        self, position, orientation, magnetization, polarization, style, **kwargs
    ):
        super().__init__(position, orientation, style=style, **kwargs)

        self._polarization = None
        self._magnetization = None
        if magnetization is not None:
            self.magnetization = magnetization
            if polarization is not None:
                raise ValueError(
                    "The attributes magnetization and polarization are dependent. "
                    "Only one can be provided at magnet initialization."
                )
        if polarization is not None:
            self.polarization = polarization

    @property
    def magnetization(self):
        """Object magnetization attribute getter and setter."""
        return self._magnetization

    @magnetization.setter
    def magnetization(self, mag):
        """Set magnetization vector, array_like, shape (3,), unit A/m."""
        self._magnetization = check_format_input_vector(
            mag,
            dims=(1,),
            shape_m1=3,
            sig_name="magnetization",
            sig_type="array_like (list, tuple, ndarray) with shape (3,)",
            allow_None=True,
            unit="A/m",
        )
        mag = self._magnetization
        if ureg is not None and isinstance(mag, ureg.Quantity):
            mag = ureg.Quantity(np.array(mag.to("A/m").m), "T")
        self._polarization = mag * (4 * np.pi * 1e-7)
        # pint Quantity does not support linalg.norm??
        if (
            np.linalg.norm(
                mag if isinstance(mag, np.ndarray) else self._magnetization.to("A/m").m
            )
            < 2000
        ):
            _deprecation_warn()

    @property
    def polarization(self):
        """Object polarization attribute getter and setter."""
        return self._polarization

    @polarization.setter
    def polarization(self, mag):
        """Set polarization vector, array_like, shape (3,), unit T."""
        self._polarization = check_format_input_vector(
            mag,
            dims=(1,),
            shape_m1=3,
            sig_name="polarization",
            sig_type="array_like (list, tuple, ndarray) with shape (3,)",
            allow_None=True,
            unit="T",
        )
        pol = self._polarization
        if ureg is not None and isinstance(pol, ureg.Quantity):
            pol = ureg.Quantity(np.array(mag.to("T").m), "A/m")
        self._magnetization = pol / (4 * np.pi * 1e-7)


class BaseCurrent(BaseSource):
    """provides scalar current attribute"""

    _style_class = CurrentStyle

    def __init__(self, position, orientation, current, style, **kwargs):
        super().__init__(position, orientation, style=style, **kwargs)
        self.current = current

    @property
    def current(self):
        """Object current attribute getter and setter."""
        return self._current

    @current.setter
    def current(self, current):
        """Set current value, scalar, unit A."""
        # input type and init check
        self._current = check_format_input_scalar(
            current,
            sig_name="current",
            sig_type="`None` or a number (int, float)",
            allow_None=True,
        )


def _deprecation_warn():
    warnings.warn(
        (
            "You have entered a very low magnetization."
            "In Magpylib v5 magnetization is given in units of A/m, "
            "while polarization is given in units of T."
        ),
        MagpylibDeprecationWarning,
        stacklevel=2,
    )
