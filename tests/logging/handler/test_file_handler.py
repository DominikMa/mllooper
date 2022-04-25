import io
import json
import logging

import numpy as np
import torch.nn
from PIL import Image

from mllooper.logging.handler import FileLog
from mllooper.logging.messages import BytesIOLogMessage, StringIOLogMessage


def test_bytes_io_log_message(tmp_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_log = FileLog(log_dir=tmp_path, time_stamp=None)

    bytes_io = io.BytesIO()
    bytes_io.write(json.dumps({'test': 42}).encode('utf-8'))

    logger.info(
        BytesIOLogMessage(bytes=bytes_io, name='testfile.json', step=0)
    )

    with open(tmp_path.joinpath('testfile-0.json'), encoding='utf-8') as f:
        test_data = json.load(f)

    assert test_data == {'test': 42}


def test_torch_bytes_io_log_message(tmp_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_log = FileLog(log_dir=tmp_path, time_stamp=None)

    bytes_io = io.BytesIO()

    torch_module = torch.nn.Linear(in_features=10, out_features=1)
    torch_state_dict = torch_module.state_dict()

    torch.save(torch_state_dict, bytes_io)

    logger.info(
        BytesIOLogMessage(bytes=bytes_io, name='torch_state_dict.pth', step=42)
    )

    loaded_torch_state_dict = torch.load(tmp_path.joinpath('torch_state_dict-42.pth'))

    for k in torch_state_dict:
        assert (torch_state_dict[k] == loaded_torch_state_dict[k]).all()


def test_pil_bytes_io_log_message(tmp_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_log = FileLog(log_dir=tmp_path, time_stamp=None)

    bytes_io = io.BytesIO()

    test_image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    Image.fromarray(test_image).save(bytes_io, format='png')

    logger.info(
        BytesIOLogMessage(bytes=bytes_io, name='test_image.png', step=None)
    )
    loaded_test_image = np.array(Image.open(tmp_path.joinpath('test_image.png')))

    assert np.all((test_image == loaded_test_image))


def test_string_io_log_message(tmp_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_log = FileLog(log_dir=tmp_path, time_stamp=None)

    string_io = io.StringIO()
    json.dump({'test': 42}, string_io)

    logger.info(
        StringIOLogMessage(text=string_io, name='testfile.json', step=0)
    )

    with open(tmp_path.joinpath('testfile-0.json'), encoding='utf-8') as f:
        test_data = json.load(f)

    assert test_data == {'test': 42}
