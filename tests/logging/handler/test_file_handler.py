import io
import json
import logging
from datetime import datetime

import numpy as np
import torch.nn
from PIL import Image

from mllooper.logging import handler
from mllooper.logging.handler import FileLog, FileLogConfig
from mllooper.logging.messages import BytesIOLogMessage, StringIOLogMessage


def test_bytes_io_log_message(tmp_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_log = FileLog(log_dir=tmp_path, log_dir_exist_ok=True)

    bytes_io = io.BytesIO()
    bytes_io.write(json.dumps({'test': 42}).encode('utf-8'))

    logger.info(
        BytesIOLogMessage(bytes=bytes_io, name='testfile.json', step=0)
    )

    with open(file_log.log_dir.joinpath('testfile-0.json'), encoding='utf-8') as f:
        test_data = json.load(f)

    assert test_data == {'test': 42}


def test_torch_bytes_io_log_message(tmp_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_log = FileLog(log_dir=tmp_path, log_dir_exist_ok=True)

    bytes_io = io.BytesIO()

    torch_module = torch.nn.Linear(in_features=10, out_features=1)
    torch_state_dict = torch_module.state_dict()

    torch.save(torch_state_dict, bytes_io)

    logger.info(
        BytesIOLogMessage(bytes=bytes_io, name='torch_state_dict.pth', step=42)
    )

    loaded_torch_state_dict = torch.load(file_log.log_dir.joinpath('torch_state_dict-42.pth'))

    for k in torch_state_dict:
        assert (torch_state_dict[k] == loaded_torch_state_dict[k]).all()


def test_pil_bytes_io_log_message(tmp_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_log = FileLog(log_dir=tmp_path, log_dir_exist_ok=True)

    bytes_io = io.BytesIO()

    test_image = np.random.randint(0, 256, size=(10, 10, 3), dtype=np.uint8)
    Image.fromarray(test_image).save(bytes_io, format='png')

    logger.info(
        BytesIOLogMessage(bytes=bytes_io, name='test_image.png', step=None)
    )
    loaded_test_image = np.array(Image.open(file_log.log_dir.joinpath('test_image.png')))

    assert np.all((test_image == loaded_test_image))


def test_string_io_log_message(tmp_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    file_log = FileLog(log_dir=tmp_path, log_dir_exist_ok=True)

    string_io = io.StringIO()
    json.dump({'test': 42}, string_io)

    logger.info(
        StringIOLogMessage(text=string_io, name='testfile.json', step=0)
    )

    with open(file_log.log_dir.joinpath('testfile-0.json'), encoding='utf-8') as f:
        test_data = json.load(f)

    assert test_data == {'test': 42}


def test_identical_file_log_timestamp(tmp_path):
    old_timestamp = handler._TIMESTAMP
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    timestamp = datetime.now().replace(microsecond=0)

    first_file_log: FileLog = FileLogConfig(log_dir=tmp_path, timestamp=timestamp).load()
    first_file_log_log_dir = first_file_log.log_dir

    first_string_io = io.StringIO()
    json.dump({'test': 'first_file_log'}, first_string_io)
    logger.info(StringIOLogMessage(text=first_string_io, name='testfile.json', step=0))
    first_file_log.teardown(state=None)

    with open(first_file_log_log_dir.joinpath('testfile-0.json'), encoding='utf-8') as f:
        test_data = json.load(f)

    assert test_data == {'test': 'first_file_log'}

    second_file_log = FileLogConfig(log_dir=tmp_path, timestamp=timestamp).load()
    second_file_log_log_dir = second_file_log.log_dir

    second_string_io = io.StringIO()
    json.dump({'test': 'second_file_log'}, second_string_io)
    logger.info(StringIOLogMessage(text=second_string_io, name='testfile.json', step=0))
    second_file_log.teardown(state=None)

    with open(second_file_log_log_dir.joinpath('testfile-0.json'), encoding='utf-8') as f:
        test_data = json.load(f)

    assert test_data == {'test': 'second_file_log'}

    # second log SHOULD overwrite first log since they are in the same process
    with open(first_file_log_log_dir.joinpath('testfile-0.json'), encoding='utf-8') as f:
        test_data = json.load(f)

    assert test_data == {'test': 'second_file_log'}
    handler._TIMESTAMP = old_timestamp


def test_identical_file_log_timestamp_two_processes(tmp_path):
    old_timestamp = handler._TIMESTAMP

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    timestamp = datetime.now().replace(microsecond=0)

    first_file_log: FileLog = FileLogConfig(log_dir=tmp_path, timestamp=timestamp).load()
    first_file_log_log_dir = first_file_log.log_dir

    first_string_io = io.StringIO()
    json.dump({'test': 'first_file_log'}, first_string_io)
    logger.info(StringIOLogMessage(text=first_string_io, name='testfile.json', step=0))
    first_file_log.teardown(state=None)

    with open(first_file_log_log_dir.joinpath('testfile-0.json'), encoding='utf-8') as f:
        test_data = json.load(f)

    assert test_data == {'test': 'first_file_log'}

    # simulate second process
    handler._TIMESTAMP = None

    second_file_log = FileLogConfig(log_dir=tmp_path, timestamp=timestamp).load()
    second_file_log_log_dir = second_file_log.log_dir

    second_string_io = io.StringIO()
    json.dump({'test': 'second_file_log'}, second_string_io)
    logger.info(StringIOLogMessage(text=second_string_io, name='testfile.json', step=0))
    second_file_log.teardown(state=None)

    with open(second_file_log_log_dir.joinpath('testfile-0.json'), encoding='utf-8') as f:
        test_data = json.load(f)

    assert test_data == {'test': 'second_file_log'}

    # second log SHOULD overwrite first log since they are in the same process
    with open(first_file_log_log_dir.joinpath('testfile-0.json'), encoding='utf-8') as f:
        test_data = json.load(f)

    assert test_data == {'test': 'first_file_log'}
    handler._TIMESTAMP = old_timestamp

