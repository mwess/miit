import json
import uuid
from dataclasses import dataclass, field
from os.path import join
from typing import Any


import numpy, numpy as np
import pandas, pandas as pd


from miit.registerers.base_registerer import Registerer, RegistrationResult
from miit.spatial_data.base_types.base_imaging import BasePointset
from miit.utils.utils import create_if_not_exists


@dataclass(kw_only=True)
class Pointset(BasePointset):

    data: pandas.core.frame.DataFrame
    _id: uuid.UUID = field(init=False)
    name: str = ''
    x_axis: Any = 'x'
    y_axis: Any = 'y'
    index_col: Any | None = None
    header: Any | None = 'infer'

    def __post_init__(self) -> None:
        self._id = uuid.uuid1()

    def apply_transform(self, registerer: Registerer, transformation: RegistrationResult, **kwargs: dict) -> Any:
        warped_pc = self.data.copy()
        pc_ = self.data[[self.x_axis, self.y_axis]].to_numpy()
        coordinates_transformed = registerer.transform_pointset(pc_, transformation, **kwargs)
        if isinstance(coordinates_transformed, np.ndarray):
            temp_df = pd.DataFrame(coordinates_transformed, columns=[self.x_axis, self.y_axis])
            coordinates_transformed = temp_df
        warped_pc = warped_pc.assign(x=coordinates_transformed[self.x_axis].values, y=coordinates_transformed[self.y_axis].values)
        return Pointset(data=warped_pc,
                        name=self.name,
                        x_axis=self.x_axis,
                        y_axis=self.y_axis,
                        index_col=self.index_col,
                        header=self.header)

    def crop(self, xmin: int, xmax: int, ymin: int, ymax: int):
        self.data[self.x_axis] = self.data[self.x_axis] - ymin
        self.data[self.y_axis] = self.data[self.y_axis] - xmin

    def resize(self, width: float, height: float):
        # Remember to convert new dimensions to scale.
        self.data[self.x_axis] = self.data[self.x_axis] * width
        self.data[self.y_axis] = self.data[self.y_axis] * height

    def rescale(self, scaling_factor: float):
        self.data[self.x_axis] = self.data[self.x_axis] * scaling_factor
        self.data[self.y_axis] = self.data[self.y_axis] * scaling_factor

    def pad(self, padding: tuple[int, int, int, int]):
        left, right, top, bottom = padding
        self.data[self.x_axis] = self.data[self.x_axis] + left
        self.data[self.y_axis] = self.data[self.y_axis] + top

    def flip(self, ref_img_shape: tuple[int, int], axis: int = 0):
        if axis == 0:
            center_x = ref_img_shape[1] // 2
            self.data.x = self.data.x + 2 * (center_x - self.data.x)
        elif axis == 1:
            center_y = ref_img_shape[0] // 2
            self.data.y = self.data.y + 2 * (center_y - self.data.y)
        else:
            pass

    def copy(self):
        return Pointset(data=self.data.copy(),
                        name=self.name,
                        x_axis=self.x_axis,
                        y_axis=self.y_axis,
                        index_col=self.index_col,
                        header=self.header)

    def to_numpy(self) -> numpy.ndarray:
        return self.data[[self.x_axis, self.y_axis]].to_numpy()

    @staticmethod
    def get_type() -> str:
        return 'pointset'

    def store(self,
              path: str):
        create_if_not_exists(path)
        fname = 'pointset.csv'
        fpath = join(path, fname)
        index = True if self.index_col is not None else False
        header = True if self.header is not None else False
        self.data.to_csv(fpath, header=header, index=index)
        attributes = {
            'name': self.name,
            'header': self.header,
            'index_col': self.index_col,
            'x_axis': self.x_axis,
            'y_axis': self.y_axis,
            'id': str(self._id)
        }
        with open(join(path, 'attributes.json'), 'w') as f:
            json.dump(attributes, f)

    @classmethod
    def load(cls,
             path: str) -> 'Pointset':
        with open(join(path, 'attributes.json')) as f:
            attributes = json.load(f)
        header = attributes['header']
        index_col = attributes['index_col']
        x_axis = attributes['x_axis']
        y_axis = attributes['y_axis']
        name = attributes['name']
        id_ = uuid.UUID(attributes['id'])
        df = pd.read_csv(join(path, 'pointset.csv'), header=header, index_col=index_col)
        ps = cls(data=df,
                 name=name,
                 header=header,
                 index_col=index_col,
                 x_axis=x_axis,
                 y_axis=y_axis)
        ps._id = id_
        return ps

    @classmethod
    def load_from_path(cls, 
                       path_to_data: str,
                       x_axis: Any = 'x',
                       y_axis: Any = 'y',
                       index_col: Any | None = None,
                       header: Any | None = 'infer',
                       name: str = '') -> 'Pointset':
        """Load a Pointset object.

        Args:
            path_to_data (str): Path to csv file.
            x_axis (Any, optional): Index used to access x axis. Defaults to 'x'.
            y_axis (Any, optional): Index used to access y axis. Defaults to 'y'.
            index_col (Optional[Any], optional): index_col argument passed on to DataFrame. Defaults to None.
            header (Optional[Any], optional): header argument passed on to DataFrame. Defaults to 'infer'.
            name (str, optional): Optional identifier. Defaults to ''.

        Returns:
            Pointset: Initialized pointset object.
        """
        data = pd.read_csv(path_to_data, index_col=index_col, header=header)
        return cls(
            data=data,
            name=name,
            x_axis=x_axis,
            y_axis=y_axis,
            index_col=index_col,
            header=header
        )