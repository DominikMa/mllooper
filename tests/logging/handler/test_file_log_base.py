import pytest

from mllooper.logging.handler import FileLogBaseConfig


def test_raise_on_wrong_extra(tmp_path):
    file_log_base_config = FileLogBaseConfig(
        log_dir=tmp_path,
        log_postfix_1="aaa",
        log_postfix="bbb",
        log_postfix2="ccc",
        asdasd="ddd",
    )
    file_log_base_config._loaded_class = dict

    with pytest.raises(ValueError):
        file_log_base_config.load()


def test_raise_on_nun_number_log_postfix(tmp_path):
    file_log_base_config = FileLogBaseConfig(
        log_dir=tmp_path,
        log_postfix_10="aaa",
        log_postfix_b="bbb",
        log_postfix_2="ccc",
        log_postfix_a="ddd",
    )
    file_log_base_config._loaded_class = dict

    with pytest.raises(ValueError):
        file_log_base_config.load()


def test_extra_log_postfix(tmp_path):
    file_log_base_config = FileLogBaseConfig(
        log_dir=tmp_path,
        log_postfix_10="aaa",
        log_postfix_2="ccc",
        log_postfix_1="bbb",
    )
    file_log_base_config._loaded_class = dict
    output = file_log_base_config.load()
    assert str(output["log_dir"]).startswith(
        str(tmp_path.joinpath("bbb", "ccc", "aaa"))
    )
