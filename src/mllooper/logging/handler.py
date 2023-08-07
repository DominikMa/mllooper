import logging
import sys
from datetime import datetime
from logging import Handler
from logging import LogRecord
from pathlib import Path
from typing import Optional

import coloredlogs
import numpy as np
import torch
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from yaloader import loads

from mllooper import Module, ModuleConfig, State
from mllooper.logging.messages import TensorBoardLogMessage, TextLogMessage, ImageLogMessage, \
    HistogramLogMessage, PointCloudLogMessage, ScalarLogMessage, FigureLogMessage, ModelGraphLogMessage, \
    ModelLogMessage, ConfigLogMessage, BytesIOLogMessage, StringIOLogMessage, TensorBoardAddCustomScalarsLogMessage

_TIMESTAMP = None


def get_not_existing_log_dir(log_dir: Path, timestamp: datetime, create_log_dir: bool = True) -> Path:
    global _TIMESTAMP

    if _TIMESTAMP is not None:
        return log_dir.joinpath(str(_TIMESTAMP).replace(' ', '-'))

    new_log_dir = log_dir.joinpath(str(timestamp).replace(' ', '-'))
    if not new_log_dir.exists():
        if create_log_dir:
            new_log_dir.mkdir(parents=True, exist_ok=False)
        _TIMESTAMP = timestamp
        return new_log_dir

    while True:
        new_timestamp = timestamp.replace(microsecond=datetime.now().microsecond)
        new_log_dir = log_dir.joinpath(str(new_timestamp).replace(' ', '-'))
        if new_log_dir.exists():
            continue
        if create_log_dir:
            new_log_dir.mkdir(parents=True, exist_ok=False)
        _TIMESTAMP = new_timestamp
        return new_log_dir


class LogHandler(Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handler: Optional[Handler] = None

    def set_handler(self, handler: Handler):
        if self.handler is None:
            self.handler = handler
            logging.getLogger().addHandler(self.handler)

    def teardown(self, state: State) -> None:
        if self.handler is not None:
            self.handler.close()
            logging.getLogger().removeHandler(self.handler)
        self.handler = None


class LogHandlerConfig(ModuleConfig):
    pass


class FileLogBase(LogHandler):
    def __init__(self, log_dir: Path, log_dir_exist_ok: bool = False, create_log_dir: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.log_dir = log_dir

        if create_log_dir:
            self.log_dir.mkdir(parents=True, exist_ok=log_dir_exist_ok)
        if not self.log_dir.exists():
            raise RuntimeError


class FileLogBaseConfig(LogHandlerConfig):
    log_dir: Path
    log_dir_exist_ok: bool = False
    create_log_dir: bool = True

    timestamp: Optional[datetime] = datetime.now().replace(microsecond=0)

    def load(self, *args, **kwargs):
        if not hasattr(self, '_loaded_class') or self._loaded_class is None:
            raise NotImplementedError

        data = dict(self)
        data.pop('timestamp')

        log_dir = self.log_dir
        if self.timestamp is None:
            if self.create_log_dir:
                log_dir.mkdir(parents=True, exist_ok=self.log_dir_exist_ok)
                data['create_log_dir'] = False
            return self._loaded_class(**data)

        if self.log_dir_exist_ok:
            log_dir = log_dir.joinpath(str(self.timestamp).replace(' ', '-'))
            data['log_dir'] = log_dir
            return self._loaded_class(**data)

        data['log_dir'] = get_not_existing_log_dir(log_dir, self.timestamp, self.create_log_dir)
        data['create_log_dir'] = False
        return self._loaded_class(**data)


class TextFileLog(FileLogBase):
    def __init__(self, level: int = logging.WARNING, **kwargs):
        super().__init__(**kwargs)
        handler = logging.FileHandler(self.log_dir.joinpath("log"))
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt=coloredlogs.DEFAULT_DATE_FORMAT
        ))
        self.set_handler(handler)


@loads(TextFileLog)
class TextFileLogConfig(FileLogBaseConfig):
    level: int = logging.WARNING


class ConsoleLog(LogHandler):
    def __init__(self, level: int = logging.WARNING, **kwargs):
        super().__init__(**kwargs)
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        handler.setFormatter(coloredlogs.ColoredFormatter(
            fmt='%(asctime)s %(name)s %(levelname)s %(message)s',
            datefmt=coloredlogs.DEFAULT_DATE_FORMAT
        ))

        self.set_handler(handler)

    def set_handler(self, handler: Handler):
        root_logger = logging.getLogger()
        if any(map(lambda handler: isinstance(handler, ConsoleLog), root_logger.handlers)):
            if self.handler is not None:
                self.handler.close()
                self.handler = None
        elif self.handler is None:
            self.handler = handler
            logging.getLogger().addHandler(self.handler)

    def teardown(self, state: State) -> None:
        pass


@loads(ConsoleLog)
class ConsoleLogConfig(LogHandlerConfig):
    level: int = logging.WARNING


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
        if isinstance(record.msg, TensorBoardAddCustomScalarsLogMessage):
            self.sw.add_custom_scalars(record.msg.layout)

        elif isinstance(record.msg, TensorBoardLogMessage):

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

        elif isinstance(record.msg, ConfigLogMessage):
            config_log: ConfigLogMessage = record.msg
            self.sw.add_text(config_log.name, text_string=config_log.formatted_text)


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
            file_path = self.log_dir.joinpath(file_name)
            torch.save(model_state_dict, file_path)

        elif isinstance(record.msg, ConfigLogMessage):
            config_log: ConfigLogMessage = record.msg

            file_name = f"{config_log.name}.yaml"
            file_path = self.log_dir.joinpath(file_name)
            file_path.write_text(config_log.config, 'utf-8')
        elif isinstance(record.msg, BytesIOLogMessage):
            bytes_log: BytesIOLogMessage = record.msg

            file_name = Path(bytes_log.name)
            if bytes_log.step is not None:
                file_name = file_name.with_stem(f"{file_name.stem}-{bytes_log.step}")
            file_path = self.log_dir.joinpath(file_name)

            file_path.write_bytes(bytes_log.bytes.getvalue())

        elif isinstance(record.msg, StringIOLogMessage):
            string_log: StringIOLogMessage = record.msg

            file_name = Path(string_log.name)
            if string_log.step is not None:
                file_name = file_name.with_stem(f"{file_name.stem}-{string_log.step}")
            file_path = self.log_dir.joinpath(file_name)

            file_path.write_text(string_log.text.getvalue(), encoding=string_log.encoding)


class TensorBoardLog(FileLogBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        handler = TensorBoardHandler(log_dir=self.log_dir)
        self.set_handler(handler)


@loads(TensorBoardLog)
class TensorBoardLogConfig(FileLogBaseConfig):
    pass


class FileLog(FileLogBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        handler = FileHandler(log_dir=self.log_dir)
        self.set_handler(handler)


@loads(FileLog)
class FileLogConfig(FileLogBaseConfig):
    pass


class MLTextFileLog(TextFileLog):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handler.addFilter(TensorBoardLogFilter())


@loads(MLTextFileLog, overwrite_tag=True)
class MLTextFileLogConfig(TextFileLogConfig):
    _yaml_tag = "!TextFileLog"


class MLConsoleLog(ConsoleLog):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.handler.addFilter(TensorBoardLogFilter())


@loads(MLConsoleLog, overwrite_tag=True)
class MLConsoleLogConfig(ConsoleLogConfig):
    _yaml_tag = "!ConsoleLog"


class TensorBoardLogFilter(logging.Filter):

    def filter(self, record):
        if isinstance(record.msg, TensorBoardLogMessage):
            return False
        return True
