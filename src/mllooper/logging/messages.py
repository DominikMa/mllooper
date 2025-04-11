from io import BytesIO, StringIO
from typing import Optional, Dict, Any, Tuple, Literal, List

import numpy as np
import torch
from matplotlib import pyplot as plt
from pydantic import validator, BaseModel, field_validator
from pydantic_core.core_schema import FieldValidationInfo
from torch import nn


class TensorBoardLogMessage(BaseModel):
    tag: Optional[str] = None
    step: Optional[int] = None

    def __str__(self):
        return self.__repr__()


class ScalarLogMessage(TensorBoardLogMessage):
    scalar: float


class TextLogMessage(TensorBoardLogMessage):
    markdown: bool = False
    text: str

    @property
    def formatted_text(self):
        text = self.text
        if not self.markdown:
            if not self.text.startswith("<pre>"):
                text = f"<pre>{text}"
            if not self.text.endswith("</pre>"):
                text = f"{text}</pre>"
        return text


class ImageLogMessage(TensorBoardLogMessage):
    """Log message for an image

    Images must be numpy uint8 array in [CxHxW] or [HxW] format.

    If `ignore_img_data` is `False` ensure that the image values are between 0 and 255 in an `np.uint8` array.
    Else just pass the image data to tensorboard like it is.
    """

    save_file: bool = False
    ignore_img_data: bool = False
    image: np.ndarray

    class Config:
        """Allow arbitrary types because `np.array` can not be checked"""

        arbitrary_types_allowed = True

    # noinspection PyArgumentList
    @field_validator("image")
    def validate_image_data(cls, image: np.ndarray, info: FieldValidationInfo):
        """Validate that image data is between 0 and 255 and as `np.uin8`"""
        ignore_img_data = info.data.get("ignore_img_data", False)
        if not ignore_img_data:
            if image.max() > 255:
                raise ValueError("Image contains values over 255")
            if image.min() < 0:
                raise ValueError("Image contains values below 0")
            if image.dtype != np.uint8:
                raise ValueError("Type of image has to be np.uint8")
        return image


class FigureLogMessage(TensorBoardLogMessage):
    """Log message for a matplotlib figure"""

    figure: plt.Figure

    class Config:
        """Allow arbitrary types because `matplotlib.pyplot.Figure` can not be checked"""

        arbitrary_types_allowed = True


class HistogramLogMessage(TensorBoardLogMessage):
    """Log message for a histogram"""

    array: np.ndarray

    class Config:
        """Allow arbitrary types because `np.array` can not be checked"""

        arbitrary_types_allowed = True

    @field_validator("array")
    def validate_points_data(cls, array: np.ndarray):
        """Validate that points data is in shape [N, 3]"""
        shape = array.shape
        if len(shape) != 1:
            raise ValueError("points array has to be of shape [N,]")
        return array


class EmbeddingsLogMessage(TensorBoardLogMessage):
    """Log message for a histogram"""

    embeddings: np.ndarray
    metadata: Optional[List[Any]] = None
    label_img: torch.Tensor = None
    metadata_header: Optional[List[str]] = None

    class Config:
        """Allow arbitrary types because `np.array` can not be checked"""

        arbitrary_types_allowed = True

    @field_validator("embeddings")
    def validate_embeddings_data(cls, array: np.ndarray):
        """Validate that points data is in shape [N, M]"""
        shape = array.shape
        if len(shape) != 2:
            raise ValueError("points array has to be of shape [N,M]")
        return array


class PointCloudLogMessage(TensorBoardLogMessage):
    points: torch.Tensor
    colors: Optional[torch.Tensor]

    class Config:
        """Allow arbitrary types because `torch.Tensor` can not be checked"""

        arbitrary_types_allowed = True

    @field_validator("points")
    def validate_points_data(cls, points: torch.Tensor):
        """Validate that points data is in shape [N, 3]"""
        shape = points.shape
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError("points array has to be of shape [N, 3]")
        return points

    @field_validator("colors")
    def validate_colors_data(
        cls, colors: Optional[torch.Tensor], info: FieldValidationInfo
    ):
        """Validate that colors data is in shape [N, 3], shame shape as points
        and data is between 0 and 255 and as `np.uin8`
        """
        if colors is None:
            return

        shape = colors.shape
        if len(shape) != 2 or shape[1] != 3:
            raise ValueError("colors array has to be of shape [N, 3]")

        points = info.data.get("points")
        if shape != points.shape:
            raise ValueError(
                "colors array and points array have to be of the same shape"
            )

        if colors.max() > 255:
            raise ValueError("colors contains values over 255")
        if colors.min() < 0:
            raise ValueError("colors contains values below 0")
        if colors.dtype != torch.uint8:
            raise ValueError(
                "Type of colors has to be torch.uint8 (use my_tensor.to(dtype=torch.uint8))"
            )
        return colors


class ModelGraphLogMessage(TensorBoardLogMessage):
    """Log message for the graph of a model"""

    model: nn.Module
    input_to_model: Any
    use_strict_trace: bool = True

    class Config:
        """Allow arbitrary types because `nn.Module` can not be checked"""

        arbitrary_types_allowed = True


class TensorBoardAddCustomScalarsLogMessage(BaseModel):
    layout: Dict[str, Dict[str, Tuple[Literal["Multiline", "Margin"], List[str]]]]


class ModelLogMessage(BaseModel):
    """Log message for a model"""

    step: Optional[int] = None
    name: str
    model: nn.Module

    class Config:
        """Allow arbitrary types because `nn.Module` can not be checked"""

        arbitrary_types_allowed = True

    def __str__(self):
        msg = f"Logged model {self.name}"
        if self.step:
            msg = f"{msg} at step {self.step}"
        return msg

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self)})"


class ConfigLogMessage(BaseModel):
    """Log message for a config"""

    name: str
    config: str

    @property
    def formatted_text(self):
        text = self.config
        if not text.startswith("<pre>"):
            text = f"<pre>{text}"
        if not text.endswith("</pre>"):
            text = f"{text}</pre>"
        return text

    def __str__(self):
        msg = f"Logged config {self.name}"
        return msg

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self)})"


class BytesIOLogMessage(BaseModel):
    """Log message for bytes"""

    step: Optional[int] = None
    name: str
    bytes: BytesIO

    class Config:
        """Allow arbitrary types because `io.BytesIO` can not be checked"""

        arbitrary_types_allowed = True

    def __str__(self):
        msg = f"Logged {self.name}"
        if self.step:
            msg = f"{msg} at step {self.step}"
        return msg

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self)})"


class StringIOLogMessage(BaseModel):
    """Log message for text"""

    step: Optional[int] = None
    name: str
    text: StringIO
    encoding: str = "utf-8"

    class Config:
        """Allow arbitrary types because `io.StringIO` can not be checked"""

        arbitrary_types_allowed = True

    def __str__(self):
        msg = f"Logged {self.name}"
        if self.step:
            msg = f"{msg} at step {self.step}"
        return msg

    def __repr__(self):
        return f"{self.__class__.__name__}({str(self)})"
