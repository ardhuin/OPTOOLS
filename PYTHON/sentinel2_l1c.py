# vim: ts=4:sts=4:sw=4
#
# @date 2019-09-27
#
# Copyright (C) 2016-2020 OceanDataLab
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Note by FA: installed glymur pyshp shapely and commented iyt import idf_converter.lib
"""
"""

import os
import numpy
import numpy.typing
import pyproj
import typing
import glymur
import logging
import datetime
import shapefile
import shapely.ops
import shapely.prepared
import shapely.geometry
import defusedxml.ElementTree
import scipy.ndimage.morphology
import scipy.interpolate
import re
import PIL.Image
import PIL.ImageDraw
#import idf_converter.lib
# import idf_converter.lib.rasterize
from idf_converter_types import InputOptions, OutputOptions, ReaderResult
from idf_converter_types import Granule

logger = logging.getLogger(__name__)

DATA_MODEL = 'GRID_YX'

SUPPORTED_VARIABLES = ['{}_TOA_reflectance', '{}_detector_index',
                       '{}_viewing_azimuth', '{}_viewing_zenith',
                       'sun_azimuth', 'sun_zenith',
                       'cloud_mask', 'land_mask',
                       '{}_atmos_reflectance']
BANDNAMES = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
             'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']


class InputPathMissing(IOError):
    """"""
    pass


class MissingLandmaskPath(Exception):
    """"""
    pass


class MissingAtmoslutPath(Exception):
    """"""
    pass


class S2VariableNotSupported(ValueError):
    """Error raised when the S-2 variable is not supported."""
    pass


class S2BandResolutionNotSupported(ValueError):
    """Error raised when the S-2 band resolution is not supported."""
    pass


class XMLNSNotFound(Exception):
    """Error raised when an XML namespace cannot be read from the XML file."""
    def __init__(self, namespace: str) -> None:
        """"""
        self.namespace = namespace


class MissingDetectorMaskPath(Exception):
    """Error raised when the detector mask file cannot be found for a given
    band."""
    def __init__(self, band_name: str) -> None:
        """"""
        self.band_name = band_name


class AngleGrid:
    """"""
    col_step: float
    row_step: float
    values: numpy.ndarray


class DetectorFootprint:
    """"""
    lowercorner: typing.List[float]
    uppercorner: typing.List[float]
    polygons: typing.List[numpy.typing.NDArray]
    detector_index: typing.List[int]


class CloudsMask:
    """"""
    lowercorner: typing.Optional[typing.List[float]]
    uppercorner: typing.Optional[typing.List[float]]
    polygons: typing.List[numpy.typing.NDArray]
    cloud_type: typing.List[int]


def help() -> typing.Tuple[str, str]:
    """Describe options supported by this reader.

    Returns
    -------
    str
        Description of the supported options
    """
    supp_vars = []
    for varname in SUPPORTED_VARIABLES:
        if varname.startswith('{}'):
            supp_vars.append(varname.format('$BAND'))
        else:
            supp_vars.append(varname)
    supp_vars_str = ', '.join(supp_vars)
    bands = ', '.join(BANDNAMES)

    inp = ('    path\tPath of the input file (S2 L1C granule directory)',
           '    variables\tComma-separated list of variable identifiers to'
           ' extract from the input file. Identifiers of supported variables'
           ' are {}. Where $BAND is the band name, ie. in'
           ' {{{}}}.'.format(supp_vars_str, bands),
           '    variable_overrides_VAR\tDefinition of variable attributes'
           ' that are either missing or erroneous in the input file.'
           ' VAR is the identifier of the variable and the value of this'
           ' option is a comma-separated list of key:value. For instance'
           ' here, variable_overrides_B02_TOA_reflectance=valid_max:0.5 allows'
           ' to directly set valid_max of B02 TOA reflectance.',
           '    resolution\tResolution in meters of the variables. Must be'
           ' a multiple of the requested band(s) native resolution.',
           '    rgb_contrast\tIf True, set automatically valid_min/valid_max'
           ' of B02, B03, B04 TOA reflectances in order to optimize RGB'
           ' contrast over ocean.',
           '    land_mask_path\tPath of OSM land polygons shapefile'
           ' (land-polygons-split-4326). Required for RGB contrast strategy'
           ' and land_mask variable.',
           '    atmos_lut_path\tPath of atmospheric reflectance LUT. Required'
           ' for RGB contrast strategy and $BAND_atmos_reflectance variable.',
           '    glymur_nthreads\tSet Glymur number of threads for JPEG2000'
           ' decoding. Default is 1.',
           )
    out = ('',)
    return ('\n'.join(inp), '\n'.join(out))


def downsample(array: numpy.ndarray, down_factor: int, operation='mean',
               dtype=None) -> numpy.ndarray:
    """
    """
    frshp = numpy.array(array.shape, dtype='int32')
    frshp -= numpy.mod(frshp, down_factor)
    dsshp = (frshp / down_factor).astype('int32')
    frsli = (slice(0, frshp[0]), slice(0, frshp[1]))
    rshp = (dsshp[0], down_factor, dsshp[1], down_factor)
    if operation == 'mean':
        # With numpy==1.21.5 and numpy.ma.MaskedArray.mean()
        # we encountered the following error:
        # -> FloatingPointError: underflow encountered in multiply
        # (division by zero ?)
        # As a consequence, we use here sum() and division.
        # (it also makes easier dtype management)
        if numpy.ma.is_masked(array):
            count = array[frsli].reshape(rshp).count(axis=(1, 3))
            array = array[frsli].reshape(rshp).sum(axis=(1, 3), dtype=dtype)
            array[count != 0] /= count[count != 0]
        else:
            array = array[frsli].reshape(rshp).sum(axis=(1, 3), dtype=dtype)
            array /= down_factor * down_factor
    elif operation == 'max':
        array = array[frsli].reshape(rshp).max(axis=(1, 3))

    return array


def rasterize_polygons_from_coords(linear_rings: list, shape: tuple,
                                   values=None, dtype=None
                                   ) -> numpy.ndarray:
    """"""
    if values is not None:
        img_type = 'L'
        if dtype is None:
            dtype = 'uint8'
    else:
        img_type = '1'
        if dtype is None:
            dtype = 'bool'
        values = [1] * len(linear_rings)
    img = PIL.Image.new(img_type, shape, 0)
    pencil = PIL.ImageDraw.Draw(img)
    for i, linear_ring in enumerate(linear_rings):
        coords = [(_[0], _[1]) for _ in linear_ring]
        pencil.polygon(coords, fill=values[i], outline=values[i])

    raster = numpy.array(img.getdata(0), dtype=dtype).reshape(shape)
    img.close()

    return raster


def read_atmos_lut(atmos_lut_path: str) -> typing.Dict[str, typing.Any]:
    """
    """
    lut = {}
    with open(atmos_lut_path, 'r') as f:
        lut_lines = f.readlines()
    lut_array = numpy.array([_.rstrip('\n').split(' ') for _ in lut_lines[2:]],
                            dtype='float32')
    lut['sun_zn'] = numpy.unique(lut_array[:, 0])
    lut['view_zn'] = numpy.unique(lut_array[:, 1])
    lut['delta_az'] = numpy.unique(lut_array[:, 2])
    lut_shape = (lut['sun_zn'].size, lut['view_zn'].size,
                 lut['delta_az'].size, lut_array.shape[1] - 3)
    lut['reflectance'] = lut_array[:, 3:].reshape(lut_shape)
    lut['bands'] = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07',
                    'B08', 'B8A', 'B09', 'B10', 'B11', 'B12']
    return lut


class Sentinel2L1CFile:
    """
    """

    NATRES = {'B01': 60, 'B02': 10, 'B03': 10, 'B04': 10,
              'B05': 20, 'B06': 20, 'B07': 20, 'B08': 10,
              'B8A': 20, 'B09': 60, 'B10': 60, 'B11': 20,
              'B12': 20}

    def __init__(self, granule_path: str, glymur_nthreads=None) -> None:
        """
        """
        self.granule_path = granule_path
        # Guess compact product naming
        # (introduced in Product Specification Document 14.0)
        granule_name = os.path.basename(granule_path)
        self._compact_naming = granule_name.startswith('L1C_T')
        # Set Glymur number of threads for JPEG2000 decoding
        if glymur_nthreads is not None:
            glymur.set_option('lib.num_threads', glymur_nthreads)
        # Read metadata XML files
        self.metadata: typing.Dict[str, typing.Any] = {}
        self._angle_grids: typing.Dict[str, AngleGrid] = {}
        self._mask_filenames: typing.Dict[str, typing.Dict[str, str]] = {}
        self._read_safe_metadata()
        self._read_granule_metadata()

    def _get_granule_path(self) -> str:
        """
        """
        return self.granule_path

    def _get_safe_path(self) -> str:
        """
        """
        granule_path = self._get_granule_path()
        return os.path.dirname(os.path.dirname(granule_path))

    def _get_granule_metadata_path(self) -> str:
        """
        """
        granule_path = self._get_granule_path()
        if self._compact_naming:
            granmtd_id = 'MTD_TL.xml'
        else:
            granule_id = os.path.basename(granule_path)
            granmtd_id = '{}.xml'.format(granule_id[:-7].replace('MSI', 'MTD'))
        return os.path.join(granule_path, granmtd_id)

    def _get_safe_metadata_path(self) -> str:
        """
        """
        safe_path = self._get_safe_path()
        if self._compact_naming:
            safemtd_id = 'MTD_MSIL1C.xml'
        else:
            safe_id = os.path.basename(safe_path)
            safemtd_id = safe_id.replace('PRD_MSI', 'MTD_SAF')
            safemtd_id = safemtd_id.replace('.SAFE', '.xml')
        return os.path.join(safe_path, safemtd_id)

    def _get_band_path(self, bandname: str) -> str:
        """
        """
        granule_path = self._get_granule_path()
        if self._compact_naming:
            safe_path = self._get_safe_path()
            safe_id = os.path.basename(safe_path)
            _, _, datatake, _, _, proj, _ = safe_id.split('_')
            band_id = '{}_{}_{}.jp2'.format(proj, datatake, bandname)
        else:
            granule_id = os.path.basename(granule_path)
            band_id = '{}_{}.jp2'.format(granule_id[:-7], bandname)
        return os.path.join(granule_path, 'IMG_DATA', band_id)

    def _get_mask_path(self,
                       mask_type: typing.Optional[str],
                       bandname: typing.Optional[str]
                       ) -> typing.Optional[str]:
        """
        """
        fnames = self._mask_filenames
        if mask_type not in fnames:
            return None
        if bandname not in fnames[mask_type]:
            return None
        safe_path = self._get_safe_path()
        fname = fnames[mask_type][bandname]
        mask_path = os.path.join(safe_path, fname)
        return mask_path

    def _read_safe_metadata(self) -> None:
        """
        """
        # Parse SAFE metadata XML
        safemtd_path = self._get_safe_metadata_path()
        tree = defusedxml.ElementTree.parse(safemtd_path)
        root = tree.getroot()
        _n1ns = re.search('{(.+?)}', root.tag)
        if _n1ns is None:
            logger.error('Could not read XML namespace "n1" from input file')
            raise XMLNSNotFound('n1')
        n1ns = _n1ns.group(1)
        xmlns = {'n1': n1ns}
        prod = root.find('n1:General_Info/Product_Info', xmlns)
        query = ['PRODUCT_START_TIME', 'PRODUCT_STOP_TIME', 'PROCESSING_LEVEL',
                 'PRODUCT_TYPE', 'PROCESSING_BASELINE', 'GENERATION_TIME']
        for q in query:
            elem = prod.find(q)
            self.metadata[elem.tag.lower()] = elem.text
        dttk = prod.find('Datatake')
        query = ['SPACECRAFT_NAME', 'DATATAKE_TYPE', 'DATATAKE_SENSING_START',
                 'SENSING_ORBIT_NUMBER', 'SENSING_ORBIT_DIRECTION']
        for q in query:
            elem = dttk.find(q)
            if q == 'SENSING_ORBIT_NUMBER':
                self.metadata[elem.tag.lower()] = int(elem.text)
            else:
                self.metadata[elem.tag.lower()] = elem.text
        image = root.find('n1:General_Info/Product_Image_Characteristics',
                          xmlns)
        for elem in image.findall('Special_Values'):
            if elem.find('SPECIAL_VALUE_TEXT').text == 'NODATA':
                int_value = int(elem.find('SPECIAL_VALUE_INDEX').text)
                self.metadata['nodata_value'] = int_value
            elif elem.find('SPECIAL_VALUE_TEXT').text == 'SATURATED':
                int_value = int(elem.find('SPECIAL_VALUE_INDEX').text)
                self.metadata['saturated_value'] = int_value
        query = ['QUANTIFICATION_VALUE']
        for q in query:
            elem = image.find(q)
            if q == 'QUANTIFICATION_VALUE':
                self.metadata[elem.tag.lower()] = float(elem.text)
            else:
                self.metadata[elem.tag.lower()] = elem.text
        elems = image.findall('Radiometric_Offset_List/RADIO_ADD_OFFSET')
        if len(elems) == 0:
            self.metadata['radio_add_offset'] = None
        else:
            # radio_add_offset to be used in reflectance formula since
            # Product Specification Document 14.9
            self.metadata['radio_add_offset'] = {}
            for elem in elems:
                bandid = int(elem.get('band_id'))
                bandname = self._band_id2name(bandid)
                float_value = float(elem.text)
                self.metadata['radio_add_offset'][bandname] = float_value
        del tree

    def _read_granule_metadata(self) -> None:
        """
        """
        # Parse granule metadata XML
        granmtd_path = self._get_granule_metadata_path()
        tree = defusedxml.ElementTree.parse(granmtd_path)
        root = tree.getroot()
        _n1ns = re.search('{(.+?)}', root.tag)
        if _n1ns is None:
            logger.error('Could not read XML namespace "n1" from input file')
            raise XMLNSNotFound('n1')
        n1ns = _n1ns.group(1)
        xmlns = {'n1': n1ns}
        gnrl = root.find('n1:General_Info', xmlns)
        query = ['TILE_ID', 'SENSING_TIME', 'Archiving_Info/ARCHIVING_CENTRE',
                 'Archiving_Info/ARCHIVING_TIME']
        for q in query:
            elem = gnrl.find(q)
            self.metadata[elem.tag.lower()] = elem.text
        geom = root.find('n1:Geometric_Info', xmlns)
        query = ['Tile_Geocoding/HORIZONTAL_CS_NAME',
                 'Tile_Geocoding/HORIZONTAL_CS_CODE']
        for q in query:
            elem = geom.find(q)
            self.metadata[elem.tag.lower()] = elem.text
        self.metadata['size'] = {}
        for elem in geom.findall('Tile_Geocoding/Size'):
            res = int(elem.get('resolution'))
            nrows = int(elem.find('NROWS').text)
            ncols = int(elem.find('NCOLS').text)
            self.metadata['size'][res] = (nrows, ncols)
        self.metadata['geoposition'] = {}
        for elem in geom.findall('Tile_Geocoding/Geoposition'):
            res = int(elem.get('resolution'))
            ulx = float(elem.find('ULX').text)
            uly = float(elem.find('ULY').text)
            xdim = float(elem.find('XDIM').text)
            ydim = float(elem.find('YDIM').text)
            self.metadata['geoposition'][res] = (ulx, uly, xdim, ydim)
        sun = geom.find('Tile_Angles/Sun_Angles_Grid')
        for anglename in ['Zenith', 'Azimuth']:
            grid_name = 'sun_{}'.format(anglename.lower())
            entry = sun.find(anglename)
            self._angle_grids[grid_name] = self._parse_angle_entry(entry)
        viewings = geom.findall('Tile_Angles/Viewing_Incidence_Angles_Grids')
        for viewing in viewings:
            bandid = int(viewing.get('bandId'))
            bandname = self._band_id2name(bandid)
            detectorid = int(viewing.get('detectorId'))
            for anglename in ['Zenith', 'Azimuth']:
                grid_name = '{}_{:02d}_viewing_{}'.format(bandname, detectorid,
                                                          anglename.lower())
                entry = viewing.find(anglename)
                pentry = self._parse_angle_entry(entry)
                if numpy.isfinite(pentry.values).any():
                    self._angle_grids[grid_name] = pentry
        qual = root.find('n1:Quality_Indicators_Info', xmlns)
        for msk in qual.findall('Pixel_Level_QI/MASK_FILENAME'):
            bandid = msk.get('bandId')
            mtype = msk.get('type')
            fname = msk.text
            masked_bandname = None
            if bandid is not None:
                masked_bandname = self._band_id2name(int(bandid))
            if mtype not in self._mask_filenames:
                self._mask_filenames[mtype] = {}
            self._mask_filenames[mtype][masked_bandname] = fname
        del tree

    # note: static method
    def _parse_angle_entry(self, entry
                           ) -> AngleGrid:
        """
        """
        col_step = float(entry.find('COL_STEP').text)
        row_step = float(entry.find('ROW_STEP').text)
        values = entry.findall('Values_List/VALUES')
        shape = (len(values), len(values[0].text.split(' ')))
        grid = numpy.zeros(shape, dtype='float32')
        for line, val in enumerate(values):
            grid[line, :] = numpy.array([float(e) for e in val.text.split(' ')])

        result = AngleGrid()
        result.col_step = col_step
        result.row_step = row_step
        result.values = grid
        return result

    def _read_detfoo_gml(self, detfoo_gml_path: str
                         ) -> DetectorFootprint:
        """
        """
        detfoo = DetectorFootprint()
        tree = defusedxml.ElementTree.parse(detfoo_gml_path)
        root = tree.getroot()
        xmlns = {'eop': 'http://www.opengis.net/eop/2.0',
                 'gml': 'http://www.opengis.net/gml/3.2'}
        # Envelope
        lowc = root.find('gml:boundedBy/gml:Envelope/gml:lowerCorner', xmlns)
        detfoo.lowercorner = [float(crd) for crd in lowc.text.split(' ')]
        upc = root.find('gml:boundedBy/gml:Envelope/gml:upperCorner', xmlns)
        detfoo.uppercorner = [float(crd) for crd in upc.text.split(' ')]
        # Detector polygons
        numpy_guess = -1
        xpath = 'eop:extentOf/gml:Polygon/gml:exterior/gml:LinearRing/gml:posList'  # noqa: E501
        polygons = []
        detector_index = []
        for feature in root.findall('eop:maskMembers/eop:MaskFeature', xmlns):
            # mask_type_node = feature.find('eop:maskType', xmlns)
            # if ((mask_type_node is None)
            #     or ('DETECTOR_FOOTPRINT' != mask_type_node.text)):
            #     continue

            _id = feature.get('{{{}}}id'.format(xmlns['gml']))
            detector_index.append(int(_id.split('-')[2]))

            coords_node = feature.find(xpath, xmlns)
            coords_count = int(coords_node.get('srsDimension'))
            _coords = [float(_) for _ in coords_node.text.split(' ')]
            _coords = numpy.array(_coords).reshape((numpy_guess, coords_count))
            polygons.append(_coords)
        detfoo.polygons = polygons
        detfoo.detector_index = detector_index
        return detfoo

    def _read_clouds_gml(self, clouds_gml_path: str
                         ) -> CloudsMask:
        """
        """
        clouds = CloudsMask()
        tree = defusedxml.ElementTree.parse(clouds_gml_path)
        root = tree.getroot()
        xmlns = {'eop': 'http://www.opengis.net/eop/2.0',
                 'gml': 'http://www.opengis.net/gml/3.2'}
        # Envelope
        lowc = root.find('gml:boundedBy/gml:Envelope/gml:lowerCorner', xmlns)
        if lowc is None:
            clouds.lowercorner = None
        else:
            clouds.lowercorner = [float(crd) for crd in lowc.text.split(' ')]
        upc = root.find('gml:boundedBy/gml:Envelope/gml:upperCorner', xmlns)
        if upc is None:
            clouds.uppercorner = None
        else:
            clouds.uppercorner = [float(crd) for crd in upc.text.split(' ')]
        # cloud polygons
        numpy_guess = -1
        xpath = 'eop:extentOf/gml:Polygon/gml:exterior/gml:LinearRing/gml:posList'  # noqa: E501
        polygons = []
        cloud_type = []  # 0=OPAQUE / 1=CIRRUS
        mask_types = ['OPAQUE', 'CIRRUS']
        for feature in root.findall('eop:maskMembers/eop:MaskFeature', xmlns):
            mask_type_node = feature.find('eop:maskType', xmlns)
            # if (mask_type_node is None) or ('OPAQUE' != mask_type_node.text):
            #     continue
            mask_type = mask_type_node.text
            cloud_type.append(mask_types.index(mask_type))

            coords_node = feature.find(xpath, xmlns)
            coords_count = int(coords_node.get('srsDimension'))
            _coords = [float(_) for _ in coords_node.text.split(' ')]
            _coords = numpy.array(_coords).reshape((numpy_guess, coords_count))
            polygons.append(_coords)
        clouds.polygons = polygons
        clouds.cloud_type = cloud_type
        return clouds

    def _get_angle_grid(self, angle_type: str, bandname=None, detector=None
                        ) -> tuple:
        """
        """
        angle_name = angle_type
        if bandname is not None and detector is not None:
            angle_name = '{}_{:02d}_{}'.format(bandname, detector, angle_name)
        if angle_name not in self._angle_grids:
            return None, None, None, None, None
        grid = self._angle_grids[angle_name]
        values = grid.values.copy()
        ny, nx = values.shape
        ulx, uly, _dx, _dy = self.get_metadata()['geoposition'][10]
        dx = numpy.abs(grid.col_step) * numpy.sign(_dx)
        dy = numpy.abs(grid.row_step) * numpy.sign(_dy)
        xctr0 = ulx
        yctr0 = uly
        isfin = numpy.isfinite(values)
        if 'viewing' in angle_name and isfin.any():
            # Viewing angles are given over each detector area with some NaN
            # around. Here we extrapolate around detector borders in order to
            # avoid NaN inside detector area after future interpolation.
            # First we fit a polynom with valid pixels
            max_deg = 3
            deg = [min([numpy.sum(isfin.sum(axis=1) > 0) - 1, max_deg]),
                   min([numpy.sum(isfin.sum(axis=0) > 0) - 1, max_deg])]
            indfin = numpy.where(isfin)
            vdr = numpy.polynomial.polynomial.polyvander2d(indfin[0],
                                                           indfin[1],
                                                           deg)
            coeff, rs, rk, s = numpy.linalg.lstsq(vdr, values[indfin],
                                                  rcond=-1)
            coeff = coeff.reshape((deg[0] + 1, deg[1] + 1))
            # Then we apply this polynom to the neighbors of valid pixels
            struct = numpy.ones((3, 3), dtype='bool')
            dilate = scipy.ndimage.morphology.binary_dilation
            dilisfin = dilate(isfin, structure=struct, border_value=0)
            ind2fill = numpy.where(~isfin & dilisfin)
            values[ind2fill] = numpy.polynomial.polynomial.polyval2d(
                ind2fill[0], ind2fill[1], coeff)
        return values, dy, dx, yctr0, xctr0

    def get_bandnames(self) -> list:
        """
        """
        return list(self.NATRES.keys())

    def get_native_resolution(self, bandname: str) -> int:
        """
        """
        return self.NATRES[bandname]

    def _band_id2name(self, bandid: int) -> str:
        """
        """
        bandnames = self.get_bandnames()
        return bandnames[bandid]

    def get_metadata(self) -> typing.Dict[str, typing.Any]:
        """
        """
        return self.metadata

    def _downsampling_factor(self, target_resolution: int,
                             native_resolution: int) -> int:
        """
        """
        tres = int(target_resolution)
        nres = int(native_resolution)
        if tres <= 0:
            raise Exception('resolution must be > 0')
        if tres % nres != 0:
            raise Exception('resolution must be multiple of native one')
        return int(tres / nres)

    def check_resolution(self, resolution: int, bandname: str) -> bool:
        """
        """
        tres = int(resolution)
        if tres <= 0:
            return False
        nres = self.get_native_resolution(bandname)
        if tres % nres != 0:
            return False
        return True

    def get_shape(self, resolution=60) -> typing.Tuple[int, int]:
        """
        """
        downfac = self._downsampling_factor(resolution, 10)
        natny, natnx = self.get_metadata()['size'][10]
        ny = int(numpy.floor(natny / downfac))
        nx = int(numpy.floor(natnx / downfac))
        return (ny, nx)

    def _get_grid_def(self, resolution: int) -> tuple:
        """
        """
        ulx, uly, xdim, ydim = self.get_metadata()['geoposition'][10]
        dx = resolution * numpy.sign(xdim)
        dy = resolution * numpy.sign(ydim)
        return ulx, uly, dx, dy

    def get_xy(self, resolution=60
               ) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """
        """
        ulx, uly, dx, dy = self._get_grid_def(resolution)
        ny, nx = self.get_shape(resolution)
        y = numpy.arange(ny, dtype='float32') * dy + dy / 2. + uly
        x = numpy.arange(nx, dtype='float32') * dx + dx / 2. + ulx
        return x, y

    def get_lonlat(self, resolution=60
                   ) -> typing.Tuple[numpy.ndarray, numpy.ndarray]:
        """
        """
        epsg = self.get_metadata()['horizontal_cs_code']
        proj = pyproj.Proj(init=epsg)
        x, y = self.get_xy(resolution)
        nx = x.size
        ny = y.size
        x = numpy.tile(x[numpy.newaxis, :], (ny, 1))
        y = numpy.tile(y[:, numpy.newaxis], (1, nx))
        lon, lat = proj(x, y, inverse=True)
        return lon, lat

    def get_detector_index(self, bandname: str, resolution=60
                           ) -> numpy.ndarray:
        """
        """
        mask_path = self._get_mask_path('MSK_DETFOO', bandname)
        if mask_path is None:
            raise MissingDetectorMaskPath(bandname)

        if mask_path.endswith('.jp2'):
            natres = self.get_native_resolution(bandname)
            downfac = self._downsampling_factor(resolution, natres)
            # Raster mask in JP2 format
            # Since Product Specification Document 14.8
            handler = glymur.Jp2k(mask_path)
            detfoo = handler[:]  # uint8
            if downfac > 1:
                i0 = int((downfac - 1) / 2.)
                detfoo = detfoo[i0::downfac, i0::downfac]
                ny, nx = self.get_shape(resolution)
                detfoo = detfoo[:ny, :nx]
        else:
            shape = self.get_shape(resolution)
            ulx, uly, dx, dy = self._get_grid_def(resolution)
            # Polygons in GML format, to be rasterized
            # Before Product Specification Document 14.8
            detfoo_gml = self._read_detfoo_gml(mask_path)
            linear_rings = []
            for poly in detfoo_gml.polygons:
                ring = numpy.stack(((poly[:, 0] - ulx) / dx,
                                    (poly[:, 1] - uly) / dy), axis=1)
                linear_rings.append(ring)
            values = detfoo_gml.detector_index
            detfoo = rasterize_polygons_from_coords(linear_rings, shape,
                                                    values)
            # import matplotlib.pyplot as plt
            # extent = [0, shape[1], 0, shape[0]]
            # plt.imshow(detfoo, origin='lower', interpolation='nearest',
            #            extent=extent)
            # for poly in linear_rings:
            #     plt.plot(poly[:, 0], poly[:, 1], '+-', color='k')
            # plt.show()
            # import pdb ; pdb.set_trace()
        return detfoo

    def _get_sun_angle(self, angle_type: str, resolution=60) -> numpy.ndarray:
        """
        """
        x, y = self.get_xy(resolution)
        val, dy, dx, yc0, xc0 = self._get_angle_grid(angle_type)
        # Bilinear interpolation
        # Initial grid may contain NaN
        fy = (y[:, numpy.newaxis] - yc0) / dy
        iy = numpy.floor(fy).astype('int32')
        iy = numpy.clip(iy, 0, val.shape[0] - 2, out=iy)
        fx = (x[numpy.newaxis, :] - xc0) / dx
        ix = numpy.floor(fx).astype('int32')
        ix = numpy.clip(ix, 0, val.shape[1] - 2, out=ix)
        wy2 = fy - iy
        wy1 = 1. - wy2
        wx2 = fx - ix
        wx1 = 1. - wx2
        angle = wx1 * (wy1 * val[iy, ix] + wy2 * val[iy + 1, ix]) + \
            wx2 * (wy1 * val[iy, ix + 1] + wy2 * val[iy + 1, ix + 1])
        angle = numpy.ma.masked_invalid(angle, copy=False)
        return angle

    def get_sun_azimuth(self, resolution=60) -> numpy.ndarray:
        """
        """
        return self._get_sun_angle('sun_azimuth', resolution)

    def get_sun_zenith(self, resolution=60) -> numpy.ndarray:
        """
        """
        return self._get_sun_angle('sun_zenith', resolution)

    def _get_viewing_angle(self, angle_type: str, bandname: str, resolution=60
                           ) -> numpy.ndarray:
        """
        """
        x, y = self.get_xy(resolution)
        det_ind = self.get_detector_index(bandname, resolution)
        hist, _ = numpy.histogram(det_ind, numpy.arange(1, 14))
        angle = numpy.ma.masked_all(det_ind.shape, dtype='float32')
        for idet in range(1, 13):
            if hist[idet - 1] == 0:
                continue
            ind = numpy.where(det_ind == idet)
            val, dy, dx, yc0, xc0 = self._get_angle_grid(angle_type,
                                                         bandname, idet)
            if val is None:
                continue
            # Bilinear interpolation
            # Initial grid may contain NaN
            fy = (y[ind[0]] - yc0) / dy
            iy = numpy.floor(fy).astype('int32')
            iy = numpy.clip(iy, 0, val.shape[0] - 2, out=iy)
            fx = (x[ind[1]] - xc0) / dx
            ix = numpy.floor(fx).astype('int32')
            ix = numpy.clip(ix, 0, val.shape[1] - 2, out=ix)
            wy2 = fy - iy
            wy1 = 1. - wy2
            wx2 = fx - ix
            wx1 = 1. - wx2
            _val = wx1 * (wy1 * val[iy, ix] + wy2 * val[iy + 1, ix]) + \
                wx2 * (wy1 * val[iy, ix + 1] + wy2 * val[iy + 1, ix + 1])
            _val = numpy.ma.masked_invalid(_val, copy=False)
            angle[ind] = _val
        return angle

    def get_viewing_azimuth(self, bandname: str, resolution=60
                            ) -> numpy.ndarray:
        """
        """
        return self._get_viewing_angle('viewing_azimuth', bandname, resolution)

    def get_viewing_zenith(self, bandname: str, resolution=60
                           ) -> numpy.ndarray:
        """
        """
        return self._get_viewing_angle('viewing_zenith', bandname, resolution)

    def get_digital_number(self, bandname: str, resolution=60
                           ) -> numpy.ndarray:
        """
        """
        natres = self.get_native_resolution(bandname)
        downfac = self._downsampling_factor(resolution, natres)
        band_path = self._get_band_path(bandname)
        handler = glymur.Jp2k(band_path)
        dnum = numpy.ma.masked_equal(handler[:], 0)  # uint16
        if downfac > 1:
            dnum = downsample(dnum, downfac, dtype='float32')
            dnum = dnum.round().astype('uint16')
        return dnum

    def get_toa_reflectance(self, bandname: str, resolution=60
                            ) -> numpy.ndarray:
        """
        """
        natres = self.get_native_resolution(bandname)
        downfac = self._downsampling_factor(resolution, natres)
        metadata = self.get_metadata()
        toar = self.get_digital_number(bandname, natres).astype('float32')
        offset = metadata['radio_add_offset']
        if offset is not None:
            toar += offset[bandname]
        toar /= metadata['quantification_value']
        if downfac > 1:
            toar = downsample(toar, downfac, dtype='float32')
        return toar

    def get_cloud_mask(self, resolution=60) -> numpy.ndarray:
        """
        """
        mask_path1 = self._get_mask_path('MSK_CLASSI', None)
        mask_path2 = self._get_mask_path('MSK_CLOUDS', None)
        if mask_path1 is not None:
            tres = int(resolution)
            if tres <= 0:
                raise Exception('resolution must be > 0')
            if tres % 60 == 0:
                upfac = 1
                downfac = self._downsampling_factor(tres, 60)
            elif tres % 30 == 0:
                upfac = 2
                downfac = self._downsampling_factor(tres, 30)
            elif tres % 20 == 0:
                upfac = 3
                downfac = self._downsampling_factor(tres, 20)
            elif tres % 10 == 0:
                upfac = 6
                downfac = self._downsampling_factor(tres, 10)
            else:
                raise Exception('resolution must be multiple of native one')
            # Raster mask in JP2 format
            # Since Product Specification Document 14.8
            # Warning: MSK_CLASSI is given at 60m resolution
            # band1 = Opaque clouds
            # band2 = Cirrus clouds
            # band3 = Snow and Ice areas
            handler = glymur.Jp2k(mask_path1)
            cloud_mask = handler[:, :, :2].sum(axis=-1) != 0
            if upfac != 1:
                iy = numpy.repeat(numpy.arange(cloud_mask.shape[0]), upfac)
                ix = numpy.repeat(numpy.arange(cloud_mask.shape[1]), upfac)
                cloud_mask = cloud_mask[iy[:, numpy.newaxis],
                                        ix[numpy.newaxis, :]]
            if downfac != 1:
                cloud_mask = downsample(cloud_mask, downfac, operation='max')
        elif mask_path2 is not None:
            shape = self.get_shape(resolution)
            ulx, uly, dx, dy = self._get_grid_def(resolution)
            # Polygons in GML format, to be rasterized
            # Before Product Specification Document 14.8
            clouds_gml = self._read_clouds_gml(mask_path2)
            linear_rings = []
            for poly in clouds_gml.polygons:
                ring = numpy.stack(((poly[:, 0] - ulx) / dx,
                                    (poly[:, 1] - uly) / dy), axis=1)
                linear_rings.append(ring)
            cloud_mask = rasterize_polygons_from_coords(linear_rings, shape)
        else:
            cloud_mask = None
        return cloud_mask

    def get_land_mask(self, land_mask_path: str, resolution=60
                      ) -> numpy.ndarray:
        """
        """
        ulx, uly, dx, dy = self._get_grid_def(resolution)
        ny, nx = self.get_shape(resolution)
        # Set x/y polygon surrounding the data
        ul_corner = (ulx, uly)
        ur_corner = (ulx + nx * dx, uly)
        br_corner = (ulx + nx * dx, uly + ny * dy)
        bl_corner = (ulx, uly + ny * dy)
        _bbox_polygon = shapely.geometry.Polygon([ul_corner, ur_corner,
                                                  br_corner, bl_corner])
        # Landmask is provided with lon/lat polygons so the bbox coordinates
        # must be reprojected
        wgs84 = pyproj.CRS('EPSG:4326')
        epsg_code = self.get_metadata()['horizontal_cs_code']
        utm = pyproj.CRS(epsg_code)
        utm_to_wgs84 = pyproj.Transformer.from_crs(utm, wgs84, always_xy=True)
        _bbox_polygon = shapely.ops.transform(utm_to_wgs84.transform,
                                              _bbox_polygon)
        # Prepare the polygon to optimize intersection test in the upcoming
        # loop
        bbox_polygon = shapely.prepared.prep(_bbox_polygon)
        # Later on the land polygons will be used in UTM coordinates, so
        # prepare the projection
        wgs84_to_utm = pyproj.Transformer.from_crs(wgs84, utm, always_xy=True)
        # Only keep landmask shapes that intersect the granule bbox
        related_shapes = []
        f = shapefile.Reader(land_mask_path)
        for shape in f.iterShapes():
            shapely_shape = shapely.geometry.shape(shape)
            # envelope = shapely_shape.envelope
            # if not bbox_polygon.intersects(envelope):
            #     continue
            if not bbox_polygon.intersects(shapely_shape):
                continue
            intersection_shape = _bbox_polygon.intersection(shapely_shape)
            utm_shape = shapely.ops.transform(wgs84_to_utm.transform,
                                              intersection_shape)
            related_shapes.append(utm_shape)
        f.close()
        # print('{} landmask polygons found'.format(len(related_shapes)))
        # import pdb ; pdb.set_trace()

        # Rasterize landmask shapes
        linear_rings = []
        for shape in related_shapes:
            if isinstance(shape, shapely.geometry.Polygon):
                x, y = numpy.array(shape.exterior.coords.xy)
                ring = numpy.stack(((x - ulx) / dx,
                                    (y - uly) / dy), axis=1)
                linear_rings.append(ring)
            elif isinstance(shape, shapely.geometry.MultiPolygon):
                for polygon in shape.geoms:
                    x, y = numpy.array(polygon.exterior.coords.xy)
                    ring = numpy.stack(((x - ulx) / dx,
                                        (y - uly) / dy), axis=1)
                    linear_rings.append(ring)
            # else:
            #     print(type(shape))
        if len(linear_rings) == 0:
            land_mask = numpy.zeros((ny, nx), dtype='bool')
        else:
            land_mask = rasterize_polygons_from_coords(linear_rings, (ny, nx))
        return land_mask

    # def get_atmos_reflectance(self, atmos_lut_path, bandname, resolution=60):
    #     """
    #     """
    #     # Get detector index and angles
    #     det_ind = self.get_detector_index(bandname, resolution)
    #     sun_az = self.get_sun_azimuth(resolution)
    #     sun_zn = self.get_sun_zenith(resolution)
    #     view_az = self.get_viewing_azimuth(bandname, resolution)
    #     view_zn = self.get_viewing_zenith(bandname, resolution)
    #     # Read LUT and set interpolator
    #     atmos_lut = read_atmos_lut(atmos_lut_path)
    #     lut_points = (atmos_lut['sun_zn'], atmos_lut['view_zn'],
    #                   atmos_lut['delta_az'])
    #     lut_iband = atmos_lut['bands'].index(bandname)
    #     lut_values = atmos_lut['reflectance'][:, :, :, lut_iband]
    #     interpolator = scipy.interpolate.RegularGridInterpolator
    #     lut_func = interpolator(lut_points, lut_values, method='linear',
    #                             bounds_error=False, fill_value=None)
    #     # Apply interpolator per detector
    #     hist, _ = numpy.histogram(det_ind, numpy.arange(1, 14))
    #     atmosr = numpy.ma.masked_all(det_ind.shape, dtype='float32')
    #     for idet in range(1, 13):
    #         if hist[idet - 1] == 0:
    #             continue
    #         ind = numpy.where(det_ind == idet)
    #         saz, szn = sun_az[ind], sun_zn[ind]
    #         vaz, vzn = view_az[ind], view_zn[ind]
    #         delta_az = numpy.abs(numpy.mod(saz - vaz + 180., 360.) - 180.)
    #         points = numpy.array((szn, vzn, delta_az)).transpose()
    #         atmosr[ind] = lut_func(points)
    #     return atmosr

    def get_atmos_reflectance(self, atmos_lut_path: str, bandname: str,
                              resolution=60) -> numpy.ndarray:
        """
        """
        # Get angles
        sun_az = self.get_sun_azimuth(resolution)
        sun_zn = self.get_sun_zenith(resolution)
        view_az = self.get_viewing_azimuth(bandname, resolution)
        view_zn = self.get_viewing_zenith(bandname, resolution)
        # Read LUT and set interpolator
        atmos_lut = read_atmos_lut(atmos_lut_path)
        lut_points = (atmos_lut['sun_zn'], atmos_lut['view_zn'],
                      atmos_lut['delta_az'])
        lut_iband = atmos_lut['bands'].index(bandname)
        lut_values = atmos_lut['reflectance'][:, :, :, lut_iband]
        interpolator = scipy.interpolate.RegularGridInterpolator
        lut_func = interpolator(lut_points, lut_values, method='linear',
                                bounds_error=False, fill_value=None)
        # Apply interpolator
        atmosr = numpy.ma.masked_all(sun_az.shape, dtype='float32')
        ind = numpy.where(~numpy.ma.getmaskarray(sun_az) &
                          ~numpy.ma.getmaskarray(sun_zn) &
                          ~numpy.ma.getmaskarray(view_az) &
                          ~numpy.ma.getmaskarray(view_zn))
        if ind[0].size == 0:
            return atmosr
        saz, szn = sun_az[ind], sun_zn[ind]
        vaz, vzn = view_az[ind], view_zn[ind]
        delta_az = numpy.abs(numpy.mod(saz - vaz + 180., 360.) - 180.)
        points = numpy.array((szn, vzn, delta_az)).transpose()
        atmosr[ind] = lut_func(points)
        return atmosr


def dilate_mask(values: numpy.ndarray, dilation_length: float,
                x_spacing: float, y_spacing: float) -> numpy.ndarray:
    """"""
    center_px = 1
    branches_px = [numpy.round(dilation_length / sp).astype('int')
                   for sp in (y_spacing, x_spacing)]
    dilation_shape = [2 * branch_px + center_px for branch_px in branches_px]
    dilation_kernel = numpy.ones(dilation_shape, dtype='bool')
    dilate = scipy.ndimage.morphology.binary_dilation
    result = dilate(values, structure=dilation_kernel)
    return result


def set_rgb_contrast(granule: Granule, sen2l1c: Sentinel2L1CFile,
                     resol: int, land_mask_path: str, atmos_lut_path: str
                     ) -> None:
    """"""
    rgb_id = ['B04_TOA_reflectance', 'B03_TOA_reflectance',
              'B02_TOA_reflectance']

    exists = []
    for var_id in rgb_id:
        vmin = granule.vars[var_id]['valid_min']
        vmax = granule.vars[var_id]['valid_max']
        exists.append(vmin is not None and vmax is not None)
    if all(exists):
        return

    # Set resolution for contrast computation
    target_ctst_resol = 80
    if resol >= target_ctst_resol:
        ctst_resol = resol
        downfac = 1
    else:
        downfac = int(numpy.round(target_ctst_resol / float(resol)))
        ctst_resol = resol * downfac

    # Get reflectance data
    rgb_array = []
    for var_id in rgb_id:
        array = granule.vars[var_id]['array']
        if downfac != 1:
            array = downsample(array, downfac, dtype='float32')
        rgb_array.append(array)

    # Set mask
    # Reflectance masks
    mask = numpy.zeros(rgb_array[0].shape, dtype='bool')
    for array in rgb_array:
        mask |= numpy.ma.getmaskarray(array)
    # Land mask
    if 'land_mask' in granule.vars:
        land_mask = granule.vars['land_mask']['array']
        if downfac != 1:
            land_mask = downsample(land_mask, downfac, operation='max')
    else:
        land_mask = sen2l1c.get_land_mask(land_mask_path, ctst_resol)
    dilation_length = 500.  # meters
    mask |= dilate_mask(land_mask, dilation_length, ctst_resol, ctst_resol)
    # Cloud mask
    if 'cloud_mask' in granule.vars:
        cloud_mask = granule.vars['cloud_mask']['array']
        if downfac != 1:
            cloud_mask = downsample(cloud_mask, downfac, operation='max')
    else:
        cloud_mask = sen2l1c.get_cloud_mask(ctst_resol)
    if cloud_mask is not None:
        dilation_length = 500.  # meters
        mask |= dilate_mask(cloud_mask, dilation_length, ctst_resol,
                            ctst_resol)
    if numpy.all(mask):
        return

    # vmin
    # vmin from data
    _vmin_data = []
    for array in rgb_array:
        pmask = (numpy.ma.getmaskarray(array) | mask)
        _vmin_data.append(numpy.percentile(numpy.ma.getdata(array[~pmask]),
                                           0.5))
    # vmin from atmos lut
    _vmin_atm = []
    for var_id in rgb_id:
        bandname = var_id[:3]
        atm_var_id = '{}_atmos_reflectance'.format(bandname)
        if atm_var_id in granule.vars:
            array = granule.vars[atm_var_id]['array']
        else:
            array = sen2l1c.get_atmos_reflectance(atmos_lut_path, bandname,
                                                  ctst_resol)
        _vmin_atm.append(array.mean())
    # final vmin
    shifts = [_v - _v_atm for _v, _v_atm in zip(_vmin_data, _vmin_atm)]
    shift = min(shifts)
    _vmin = [_v_atm + shift for _v_atm in _vmin_atm]

    # vmax
    slope_threshold = -40.
    _vmax_data = []
    bins = numpy.linspace(0, .5, num=51)
    for array in rgb_array:
        hist, edges = numpy.histogram(array[~mask], bins=bins)
        dbin = edges[1] - edges[0]
        hist = hist.astype('float') / hist.sum() / dbin
        slope = (hist[1:] - hist[:-1]) / dbin
        slope_ctrs = edges[1:-1]
        ind = numpy.where(slope <= slope_threshold)[0]
        if ind.size != 0:
            ind = ind.max()
            dy = slope[ind + 1] - slope[ind]
            dx = slope_ctrs[ind + 1] - slope_ctrs[ind]
            _vm = slope_ctrs[ind] + (slope_threshold - slope[ind]) / dy * dx
            _vmax_data.append(_vm)
        else:
            pmask = (numpy.ma.getmaskarray(array) | mask)
            _vmax_data.append(numpy.percentile(numpy.ma.getdata(array[~pmask]),
                                               99.5))
    vrans = [vma - vmi for vma, vmi in zip(_vmax_data, _vmin)]
    vran = max(vrans)
    _vmax = [vmi + vran for vmi in _vmin]

    # Fill
    for ibnd, var_id in enumerate(rgb_id):
        vmin = granule.vars[var_id]['valid_min']
        if vmin is None:
            granule.vars[var_id]['valid_min'] = _vmin[ibnd]
        vmax = granule.vars[var_id]['valid_max']
        if vmax is None:
            granule.vars[var_id]['valid_max'] = _vmax[ibnd]


def compute_minmax(vars_id: typing.List[str],
                   granule: Granule
                   ) -> None:
    """"""
    for var_id in vars_id:
        if var_id.endswith('_TOA_reflectance'):
            vmin = granule.vars[var_id]['valid_min']
            vmax = granule.vars[var_id]['valid_max']
            if vmin is None or vmax is None:
                values = numpy.ma.compressed(granule.vars[var_id]['array'])
                if vmin is None:
                    _vmin = numpy.percentile(values, 0.1)
                    granule.vars[var_id]['valid_min'] = _vmin
                if vmax is None:
                    _vmax = numpy.percentile(values, 99.9)
                    granule.vars[var_id]['valid_max'] = _vmax
        elif var_id.endswith('_detector_index'):
            vmin = granule.vars[var_id]['valid_min']
            vmax = granule.vars[var_id]['valid_max']
            if vmin is None:
                granule.vars[var_id]['valid_min'] = 0
            if vmax is None:
                granule.vars[var_id]['valid_max'] = 254
        elif var_id.endswith('_azimuth') or var_id.endswith('_zenith'):
            vmin = granule.vars[var_id]['valid_min']
            vmax = granule.vars[var_id]['valid_max']
            if vmin is None:
                _vmin = granule.vars[var_id]['array'].min()
                granule.vars[var_id]['valid_min'] = _vmin
            if vmax is None:
                _vmax = granule.vars[var_id]['array'].max()
                granule.vars[var_id]['valid_max'] = _vmax
        elif var_id in ['cloud_mask', 'land_mask']:
            vmin = granule.vars[var_id]['valid_min']
            vmax = granule.vars[var_id]['valid_max']
            if vmin is None:
                granule.vars[var_id]['valid_min'] = 0
            if vmax is None:
                granule.vars[var_id]['valid_max'] = 254
        elif var_id.endswith('_atmos_reflectance'):
            vmin = granule.vars[var_id]['valid_min']
            vmax = granule.vars[var_id]['valid_max']
            if vmin is None:
                _vmin = granule.vars[var_id]['array'].min()
                granule.vars[var_id]['valid_min'] = _vmin
            if vmax is None:
                _vmax = granule.vars[var_id]['array'].max()
                granule.vars[var_id]['valid_max'] = _vmax


def init_variables(vars_id: typing.List[str]
                   ) -> typing.Dict[str, typing.Dict[str, typing.Any]]:
    """"""
    variables: typing.Dict[str, typing.Dict[str, typing.Any]] = {}
    for var_id in vars_id:
        if var_id.endswith('_TOA_reflectance'):
            bandname = var_id[:3]
            long_name = '{} top-of-atmosphere reflectance'.format(bandname)
            units = None  # dimensionless ?
        elif var_id.endswith('_detector_index'):
            bandname = var_id[:3]
            long_name = '{} detector index'.format(bandname)
            units = None
        elif var_id.endswith('_viewing_azimuth'):
            bandname = var_id[:3]
            long_name = '{} viewing azimuth angle'.format(bandname)
            units = 'degree'
        elif var_id.endswith('_viewing_zenith'):
            bandname = var_id[:3]
            long_name = '{} viewing zenith angle'.format(bandname)
            units = 'degree'
        elif var_id == 'sun_azimuth':
            long_name = 'sun azimuth angle'
            units = 'degree'
        elif var_id == 'sun_zenith':
            long_name = 'sun zenith angle'
            units = 'degree'
        elif var_id == 'cloud_mask':
            long_name = 'cloud mask'
            units = None
        elif var_id == 'land_mask':
            long_name = 'land mask'
            units = None
        elif var_id.endswith('_atmos_reflectance'):
            bandname = var_id[:3]
            long_name = '{} atmospheric reflectance'.format(bandname)
            units = None
        else:
            raise S2VariableNotSupported()
        variables[var_id] = {'name': var_id,
                             'long_name': long_name,
                             'valid_min': None,
                             'valid_max': None,
                             'options': {}}
        if units is not None:
            variables[var_id]['units'] = units
    return variables


def read_data(input_opts: InputOptions,
              output_opts: OutputOptions
              ) -> typing.Iterator[ReaderResult]:
    """Read input file, extract data and metadata, store them in a Granule
    object then prepare formatting instructions to finalize the conversion to
    IDF format.

    Parameters
    ----------
    input_opts: dict
        Options and information related to input data
    output_opts: dict
        Options and information related to formatting and serialization to IDF
        format

    Returns
    -------
    tuple(dict, dict, idf_converter.lib.Granule, list)
        A tuple which contains four elements:

        - the input_options :obj:dict passed to this method

        - the output_options :obj:dict passed to this method

        - the :obj:`idf_converter.lib.Granule` where the extracted information
          has been stored

        - a :obj:list of :obj:dict describing the formatting operations that
          the converter must perform before serializing the result in IDF
          format
    """
    # Get inputs
    _input_path = input_opts.get('path', None)
    if _input_path is None:
        raise InputPathMissing()
    input_path = os.path.normpath(_input_path)

    _vars_id = input_opts.get('variables', None)
    if _vars_id is None:
        # Default
        vars_id = ['B02_TOA_reflectance',
                   'B03_TOA_reflectance',
                   'B04_TOA_reflectance']
    else:
        vars_id = _vars_id.split(',')

    _resol = input_opts.get('resolution', None)
    if _resol is None:
        # Default (will work for any band)
        resol = 60
    else:
        resol = int(_resol)

    _rgb_ctst = input_opts.get('rgb_contrast', 'False')
    rgb_contrast = _rgb_ctst in ['True', 'true']

    # OSM: land-polygons-split-4326/land_polygons.shp
    land_mask_path = input_opts.get('land_mask_path', None)

    atmos_lut_path = input_opts.get('atmos_lut_path', None)

    glymur_nthreads = input_opts.get('glymur_nthreads', None)
    if glymur_nthreads is not None:
        glymur_nthreads = int(glymur_nthreads)

    # Check inputs
    # Check vars_id
    supp_vars_all = []
    for varname in SUPPORTED_VARIABLES:
        if varname.startswith('{}'):
            for bandname in BANDNAMES:
                supp_vars_all.append(varname.format(bandname))
        else:
            supp_vars_all.append(varname)
    for var_id in vars_id:
        if var_id not in supp_vars_all:
            raise S2VariableNotSupported()
    # Disable rgb_contrast if RGB bands are not all requested
    if rgb_contrast:
        has_rgb = []
        for bandname in ['B02', 'B03', 'B04']:
            _var_id = '{}_TOA_reflectance'.format(bandname)
            has_rgb.append(_var_id in vars_id)
        rgb_contrast = all(has_rgb)
    # Check land mask path
    asking_land = 'land_mask' in vars_id
    if rgb_contrast or asking_land:
        if land_mask_path is None:
            raise MissingLandmaskPath()
    # Check atmos lut path
    asking_atmos = any([v.endswith('_atmos_reflectance') for v in vars_id])
    if rgb_contrast or asking_atmos:
        if atmos_lut_path is None:
            raise MissingAtmoslutPath()
    # Init Sentinel2L1CFile
    sen2l1c = Sentinel2L1CFile(input_path, glymur_nthreads=glymur_nthreads)
    # Check resolution
    for var_id in vars_id:
        _bandname = var_id[:3]
        if _bandname in BANDNAMES:
            if not sen2l1c.check_resolution(resol, _bandname):
                raise S2BandResolutionNotSupported()

    # Init granule
    variables = init_variables(vars_id)

    idf_version = output_opts.get('idf_version', '1.0')
    granule = idf_converter.lib.create_granule(idf_version, DATA_MODEL)
    granule.vars = variables
    idf_converter.lib.apply_var_overrides(input_opts, granule)

    # Read variables
    for var_id in vars_id:
        if var_id.endswith('_TOA_reflectance'):
            bandname = var_id[:3]
            array = sen2l1c.get_toa_reflectance(bandname, resol)
        elif var_id.endswith('_detector_index'):
            bandname = var_id[:3]
            array = sen2l1c.get_detector_index(bandname, resol)
        elif var_id.endswith('_viewing_azimuth'):
            bandname = var_id[:3]
            array = sen2l1c.get_viewing_azimuth(bandname, resol)
        elif var_id.endswith('_viewing_zenith'):
            bandname = var_id[:3]
            array = sen2l1c.get_viewing_zenith(bandname, resol)
        elif var_id == 'sun_azimuth':
            array = sen2l1c.get_sun_azimuth(resol)
        elif var_id == 'sun_zenith':
            array = sen2l1c.get_sun_zenith(resol)
        elif var_id == 'cloud_mask':
            array = sen2l1c.get_cloud_mask(resol)
        elif var_id == 'land_mask':
            array = sen2l1c.get_land_mask(land_mask_path, resol)
        elif var_id.endswith('_atmos_reflectance'):
            bandname = var_id[:3]
            array = sen2l1c.get_atmos_reflectance(atmos_lut_path, bandname,
                                                  resol)
        granule.vars[var_id]['array'] = array

    # Set valid_min / valid_max
    if rgb_contrast:
        set_rgb_contrast(granule, sen2l1c, resol, land_mask_path,
                         atmos_lut_path)

    compute_minmax(vars_id, granule)

    # Geolocation
    ny, nx = sen2l1c.get_shape(resol)
    granule.dims['y'] = ny
    granule.dims['x'] = nx
    x_values, y_values = sen2l1c.get_xy(resol)
    granule.vars['y'] = {'array': y_values,
                         'datatype': y_values.dtype,
                         'options': {}}
    granule.vars['x'] = {'array': x_values,
                         'datatype': x_values.dtype,
                         'options': {}}
    output_opts['gcp_spacing'] = int(25000 / resol)
    metadata = sen2l1c.get_metadata()
    epsg = metadata['horizontal_cs_code']
    input_opts['projection'] = epsg

    # Metadata
    s2_granule_root = os.path.dirname(input_path)
    s2_granule_count = 0
    for bname in os.listdir(s2_granule_root):
        path = os.path.join(s2_granule_root, bname)
        if os.path.isdir(path):
            s2_granule_count += 1
    if s2_granule_count > 1:
        # Multitile product (old products at beginning of S2A mission start)
        s2_granule_name = os.path.basename(input_path)
        idf_granule_name = '{}-{}m'.format(s2_granule_name, resol)
    else:
        # Nomical case: 1 granule (tile) per S2 product
        s2_safe_root = os.path.dirname(s2_granule_root)
        s2_safe_name = os.path.splitext(os.path.basename(s2_safe_root))[0]
        idf_granule_name = '{}-{}m'.format(s2_safe_name, resol)

    time_fmt = '%Y-%m-%dT%H:%M:%S.%fZ'  # eg. 2022-03-25T11:21:09.024Z
    start_time = metadata['product_start_time']
    start_dt = datetime.datetime.strptime(start_time, time_fmt)
    stop_time = metadata['product_stop_time']
    stop_dt = datetime.datetime.strptime(stop_time, time_fmt)
    platform = metadata['spacecraft_name']

    granule.meta['idf_granule_id'] = idf_granule_name
    granule.meta['idf_subsampling_factor'] = 0
    granule.meta['idf_spatial_resolution'] = resol
    granule.meta['idf_spatial_resolution_units'] = 'm'
    granule.meta['time_coverage_start'] = start_dt
    granule.meta['time_coverage_end'] = stop_dt
    granule.meta['institution'] = 'ESA'
    granule.meta['platform'] = platform
    granule.meta['platform_type'] = 'leo satellite'
    granule.meta['sensor'] = 'MSI'
    granule.meta['sensor_type'] = 'multi-spectral imager'

    # Add transforms
    transforms = []

    # Artificially increase the range on a linear scale, thus allowing
    # users to desaturate the rendering of this channel in SEAScope
    # _max_linear = _min_linear + 2 * (_max_linear - _min_linear)
    if rgb_contrast:
        rgb_vars = ['B02_TOA_reflectance', 'B03_TOA_reflectance',
                    'B04_TOA_reflectance']
        transforms.append(('stretch_extent', {'targets': rgb_vars,
                                              'stretch_factor': 2.0,
                                              'change_vmin': False,
                                              'change_vmax': True}))

    output_opts['__export'] = vars_id

    yield (input_opts, output_opts, granule, transforms)
