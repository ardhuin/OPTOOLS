# vim: ts=4:sts=4:sw=4
#
# @date 2019-09-25
#
# Copyright (C) 2016-2021 OceanDataLab
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

import numpy
import typing
import logging
import dateutil
import datetime

logger = logging.getLogger(__name__)


class Granule(object):
    """"""
    def __init__(self, idf_version: str, data_model: str,
                 dimensions: typing.List[str]) -> None:
        """"""
        self.idf_version: str = idf_version
        self.vars: typing.Dict[str, typing.Any] = {}
        self.dims: typing.Dict[str, int] = {}
        self.meta: typing.Dict[str, typing.Any] = {}
        self.data_model: str = data_model
        self._data_model_dimensions = dimensions
        for dim in dimensions:
            self.dims[dim] = 0

    def get_data_model_dimensions(self) -> typing.List[str]:
        """"""
        return self._data_model_dimensions.copy()

    def validate(self) -> bool:
        """"""
        return False


InputOptions = typing.Dict[str, typing.Any]
OutputOptions = typing.Dict[str, typing.Any]
FormatterJob = typing.Tuple[str, typing.Dict[str, typing.Any]]
AtmCorrData = typing.Dict[str, typing.Any]
NumValuesDict = typing.Dict[str, numpy.ndarray]
Extrema = typing.Tuple[float, float]
Duration = dateutil.relativedelta.relativedelta
RelativeBound = typing.Union[str, float]
CoverageInfo = typing.Tuple[typing.Optional[datetime.datetime],
                            typing.Optional[datetime.datetime],
                            typing.Optional[RelativeBound],
                            typing.Optional[RelativeBound]]
Coverage = typing.Tuple[typing.Optional[datetime.datetime],
                        typing.Optional[datetime.datetime]]
ReaderResult = typing.Tuple[typing.Dict[str, typing.Any],
                            typing.Dict[str, typing.Any],
                            Granule,
                            typing.List[FormatterJob]]
FormatterResult = typing.Tuple[InputOptions, OutputOptions, Granule]
ProcessedLists = typing.Tuple[typing.Iterable[str],
                              typing.Iterable[str],
                              typing.Iterable[str]]
TransformsList = typing.List[typing.Tuple[str, typing.Dict[str, typing.Any]]]
IdxSlices = typing.List[slice]
