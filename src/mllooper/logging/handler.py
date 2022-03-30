import logging
from logging import Handler, LogRecord
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from baselooper.logging.handler import FileLogBase, FileLogBaseConfig, TextFileLog, ConsoleLog, TextFileLogConfig, \
    ConsoleLogConfig
from torch.utils.tensorboard import SummaryWriter

from mllooper.logging.messages import TensorBoardLogMessage, TextLogMessage, ImageLogMessage, \
    HistogramLogMessage, PointCloudLogMessage, ScalarLogMessage, FigureLogMessage, ModelGraphLogMessage, ModelLogMessage


class TensorBoardHandler(Handler):
    def __init__(self, log_dir: Path):
        super().__init__()
        self.log_dir = log_dir
        # noinspection PyTypeChecker
        self.sw: SummaryWriter = SummaryWriter(log_dir=self.log_dir)

    def close(self) -> None:
        self.sw.close()
        super(TensorBoardHandler, self).close()

    def emit(self, record: LogRecord) -> None:
        # Skip if it isn't a subclass of `LogMessage`
        if not isinstance(record.msg, TensorBoardLogMessage):
            return

        tag = record.msg.tag if record.msg.tag else record.name
        step = record.msg.step if record.msg.step else 0
        if isinstance(record.msg, ScalarLogMessage):
            scalar_log: ScalarLogMessage = record.msg
            self.sw.add_scalar(tag, scalar_value=scalar_log.scalar, new_style=True, global_step=step)

        elif isinstance(record.msg, TextLogMessage):
            text_log: TextLogMessage = record.msg
            self.sw.add_text(tag, text_string=text_log.formatted_text, global_step=step)

        elif isinstance(record.msg, ImageLogMessage):
            img_log: ImageLogMessage = record.msg
            self.sw.add_image(tag, img_tensor=img_log.image, global_step=step)

        elif isinstance(record.msg, FigureLogMessage):
            figure_log: FigureLogMessage = record.msg
            self.sw.add_figure(tag, figure=figure_log.figure, global_step=step)

        elif isinstance(record.msg, HistogramLogMessage):
            hist_log: HistogramLogMessage = record.msg
            self.sw.add_histogram(tag, values=hist_log.array, global_step=step)

        elif isinstance(record.msg, PointCloudLogMessage):
            point_cloud_log: PointCloudLogMessage = record.msg

            vertices = torch.unsqueeze(point_cloud_log.points, dim=0)
            if point_cloud_log.colors is not None:
                colors = torch.unsqueeze(point_cloud_log.colors, dim=0)
            else:
                colors = None

            self.sw.add_mesh(tag, vertices=vertices, colors=colors, global_step=step)

        elif isinstance(record.msg, ModelGraphLogMessage):
            model_graph_log: ModelGraphLogMessage = record.msg
            self.sw.add_graph(model=model_graph_log.model, input_to_model=model_graph_log.input_to_model)


class FileHandler(Handler):
    def __init__(self, log_dir: Path):
        super().__init__()
        self.log_dir = log_dir

    def emit(self, record: LogRecord) -> None:
        # Skip if it isn't a subclass of `LogMessage`
        if isinstance(record.msg, ImageLogMessage) and record.msg.save_file:
            img_log: ImageLogMessage = record.msg
            tag = img_log.tag if img_log.tag else record.name
            tag = tag.replace('/', '-').replace('.', '-')

            step = img_log.step
            image = img_log.image

            # Turn C,H,W into H,W,C for PIL
            if len(image.shape) == 3:
                image = np.transpose(image, (1, 2, 0))
            file_name = f"{tag}-{step}.png" if step is not None else f"{tag}.png"
            file_path = self.log_dir.joinpath(file_name)

            image = Image.fromarray(image)
            image.save(file_path)

        elif isinstance(record.msg, ModelLogMessage):
            model_log: ModelLogMessage = record.msg

            model_state_dict = model_log.model.state_dict()
            file_name = f"{model_log.name}-{ model_log.step}.pth" if model_log.step is not None else f"{model_log.name}.pth"
            torch.save(model_state_dict, file_name)


class TensorBoardLog(FileLogBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        handler = TensorBoardHandler(log_dir=self.log_dir)
        self.set_handler(handler)


class TensorBoardLogConfig(FileLogBaseConfig, loaded_class=TensorBoardLog):
    pass


class FileLog(FileLogBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        handler = FileHandler(log_dir=self.log_dir)
        self.set_handler(handler)


class FileLogConfig(FileLogBaseConfig, loaded_class=FileLog):
    pass


class MLTextFileLog(TextFileLog):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handler.addFilter(TensorBoardLogFilter())


class MLTextFileLogConfig(TextFileLogConfig, overwrite_tag=True, loaded_class=MLTextFileLog):
    _yaml_tag = "!TextFileLog"


class MLConsoleLog(ConsoleLog):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handler.addFilter(TensorBoardLogFilter())


class MLConsoleLogConfig(ConsoleLogConfig, overwrite_tag=True, loaded_class=MLConsoleLog):
    _yaml_tag = "!ConsoleLog"


class TensorBoardLogFilter(logging.Filter):

    def filter(self, record):
        if isinstance(record.msg, TensorBoardLogMessage):
            return False
        return True
