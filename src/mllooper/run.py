import importlib
import json
import logging
import re
import subprocess
import sys
from importlib.metadata import distributions
from importlib.util import spec_from_file_location, module_from_spec, find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Tuple

import click
import git
import yaml
from click import BadParameter
from pydantic import ValidationError
from yaloader import ConfigLoader, YAMLConfigDumper
from yaml import MarkedYAMLError

from mllooper import Module, ModuleConfig
from mllooper.logging.handler import BufferingLogHandler
from mllooper.logging.messages import ConfigLogMessage
from mllooper.utils import git_get_url_rev_and_auth

TEMP_DIR = TemporaryDirectory(prefix='mllooper_tmp_')

logger = logging.getLogger('mllooper.cli')


def install_package(package_name: str):
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--force-reinstall', package_name])
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Could not install {package_name}: {e}")
    else:
        logger.info(
            f"Installed package {package_name}"
        )


def is_valid_module_name(module_name: str):
    pattern = r"^[a-zA-Z_][a-zA-Z0-9_]*$"
    return re.fullmatch(pattern, module_name)


def import_as_known_module(module_name: str):
    if not is_valid_module_name(module_name):
        raise ModuleNotFoundError
    importlib.import_module(module_name)


def import_from_disk(module_name: str):
    module_path = Path(module_name).absolute()
    if module_path.is_file() and module_path.suffix == '.py':
        name = module_path.parent.name
        location = module_path
    elif module_path.is_dir() and module_path.joinpath('__init__.py').is_file():
        name = module_path.name
        location = module_path.joinpath('__init__.py')
    else:
        raise ModuleNotFoundError

    if find_spec(name) is not None:
        name = f"{name}_mllooper_auto_import"

    spec = spec_from_file_location(name, location)
    module = module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)


def import_module(module_name: str):
    # try to import as a known module
    try:
        import_as_known_module(module_name)
    except ModuleNotFoundError as error:
        if hasattr(error, 'name') and error.name is not None and error.name != module_name:
            raise
    else:
        logger.info(
            f"Imported module {module_name}"
        )
        return

    # try to import as file or directory
    try:
        import_from_disk(module_name)
    except ModuleNotFoundError as error:
        if hasattr(error, 'name') and error.name is not None and error.name != module_name:
            raise
    else:
        logger.info(
            f"Imported module {module_name}"
        )
        return

    raise ModuleNotFoundError(f"Could not import {module_name}")


def git_import_module(module_git_url: str):
    url, rev, user_pass = git_get_url_rev_and_auth(f'git+{module_git_url}')
    name = url.split('/')[-1].split('.')[0]

    try:
        rev, path = rev.split(':', 1)
    except ValueError:
        path = ''
    rev = None if rev == '' else rev

    clone_path = TemporaryDirectory(prefix=f"{name}_", dir=TEMP_DIR.name)
    if rev is not None:
        bare_repo = git.Repo.init(clone_path.name, bare=False)
        origin = bare_repo.create_remote("origin", url=url)
        origin.fetch(
            refspec=rev,
            depth=1
        )
        bare_repo.git.checkout(rev)
        ref = bare_repo.head.ref.name
        commit = bare_repo.head.commit.hexsha
    else:
        repo = git.Repo.clone_from(
            url=url,
            to_path=clone_path.name,
            depth=1
        )
        ref = repo.head.ref.name
        commit = repo.head.commit.hexsha

    import_path = Path(clone_path.name).joinpath(path)

    # try to import as file or directory
    try:
        import_from_disk(str(import_path))
        logger.info(
            f"Imported module {name}{'' if not path else ' at ' + path} from {url} at revision {ref} ({commit})"
        )
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError(f"Could not import {module_git_url}: {error}") from error


def load_config(config_loader: ConfigLoader, run_config: str, final: bool = True):
    if (path := Path(run_config)).is_file() or Path(run_config).with_suffix('.yaml').is_file():
        try:
            constructed_run = config_loader.construct_from_file(path, final=final)
        except (FileNotFoundError, MarkedYAMLError, ValidationError) as e:
            raise BadParameter(f"{e}") from e
    else:
        try:
            constructed_run = config_loader.construct_from_string(run_config, final=final)
        except (MarkedYAMLError, ValidationError) as e:
            raise BadParameter(f"{e}") from e
    return constructed_run


@click.group()
@click.option("-c", "--config", "config_paths", multiple=True, default=[], type=Path)
@click.option("-d", "--dir", "config_dirs", multiple=True, default=[], type=Path)
@click.option("-y", "--yaml", "yaml_strings", multiple=True, default=[], type=str)
@click.option("--install", "install_packages", multiple=True, default=[])
@click.option("-i", "--import", "import_modules", multiple=True, default=[])
@click.option("-g", "--git-import", "git_import_modules", multiple=True, default=[])
@click.option("-v", "--verbose", count=True, default=0)
@click.option("--quiet", count=True, default=0)
@click.option("--global-log-level", type=int, default=30)
@click.pass_context
def cli(
        ctx,
        config_paths: Tuple[Path],
        config_dirs: Tuple[Path],
        yaml_strings: Tuple[str],
        install_packages: Tuple[str],
        import_modules: Tuple[str],
        git_import_modules: Tuple[str],
        verbose: int,
        quiet: int,
        global_log_level: int
):
    ctx.ensure_object(dict)

    logging.getLogger().setLevel(global_log_level)
    log_level = 20 - verbose * 10 + quiet * 10
    logger.setLevel(log_level)

    buffering_log_handler = BufferingLogHandler()
    logging.getLogger().addHandler(buffering_log_handler)

    # install packages before importing modules
    for package in install_packages:
        try:
            install_package(package)
        except RuntimeError as e:
            raise BadParameter(f"{e}") from e

    # import modules before creating the loader
    for module_name in import_modules:
        try:
            import_module(module_name)
        except ModuleNotFoundError as e:
            raise BadParameter(f"{e}") from e

    # import modules before creating the loader
    for module_git_url in git_import_modules:
        try:
            git_import_module(module_git_url)
        except ModuleNotFoundError as e:
            raise BadParameter(f"{e}") from e

    config_loader = ConfigLoader()

    # add configurations
    for config_dir in config_dirs:
        try:
            config_loader.load_directory(config_dir.absolute())
        except (NotADirectoryError, MarkedYAMLError, ValidationError) as e:
            raise BadParameter(f"{e}") from e

    for config_path in config_paths:
        try:
            config_loader.load_file(config_path.absolute())
        except (FileNotFoundError, MarkedYAMLError, ValidationError) as e:
            raise BadParameter(f"{e}") from e

    for yaml_string in yaml_strings:
        try:
            # config_loader.load_string(yaml_string)
            config_loader.add_single_config_string(yaml_string, priority=100)
        except (MarkedYAMLError, ValidationError) as e:
            raise BadParameter(f"{e}") from e

    logging.getLogger().removeHandler(buffering_log_handler)

    ctx.obj['config_loader'] = config_loader
    ctx.obj['buffering_log_handler'] = buffering_log_handler


@cli.command()
@click.argument('run_config', type=str)
@click.pass_obj
def run(ctx_object, run_config: str):
    config_loader = ctx_object['config_loader']
    buffering_log_handler = ctx_object['buffering_log_handler']

    previous_handlers = logging.getLogger().handlers.copy()

    # load and run the run configuration
    constructed_run = load_config(config_loader, run_config, final=True)

    if not isinstance(constructed_run, ModuleConfig):
        raise BadParameter(f"The run configuration RUN_CONFIG has to be a mllooper ModuleConfig. "
                           f"Got {type(constructed_run)} instead.")
    loaded_run = constructed_run.load()

    new_handlers = [handler for handler in logging.getLogger().handlers if handler not in previous_handlers]
    buffering_log_handler.set_targets(new_handlers)
    buffering_log_handler.flush()
    buffering_log_handler.close()

    installed_packages = ', '.join(sorted([f"{package.name}=={package.version}" for package in distributions()], key=str.lower))
    logger.info(f"Installed packages:\n{installed_packages}")

    # Log config
    YAMLConfigDumper.exclude_unset = False
    YAMLConfigDumper.exclude_defaults = False
    config = yaml.dump(constructed_run, Dumper=YAMLConfigDumper, sort_keys=False)
    logger.info(ConfigLogMessage(name='full_config', config=config))
    logger.debug(f"Full Config:\n{config}")
    YAMLConfigDumper.exclude_unset = True
    YAMLConfigDumper.exclude_defaults = True
    config = yaml.dump(constructed_run, Dumper=YAMLConfigDumper, sort_keys=False)
    logger.info(ConfigLogMessage(name='config', config=config))

    loaded_run.run()


@cli.command()
@click.argument('config', type=str)
@click.pass_obj
def build(ctx_object, config: str):
    config_loader: ConfigLoader = ctx_object['config_loader']
    buffering_log_handler: BufferingLogHandler = ctx_object['buffering_log_handler']
    buffering_log_handler.close()

    config = load_config(config_loader, config, auto_load=False, final=False)
    print(yaml.dump(config, Dumper=YAMLConfigDumper, sort_keys=False))


@cli.command()
@click.argument('tag', type=str)
@click.option("--definitions/--no-definitions", "definitions", default=False)
@click.pass_obj
def explain(ctx_object, tag: str, definitions: bool):
    config_loader: ConfigLoader = ctx_object['config_loader']
    buffering_log_handler: BufferingLogHandler = ctx_object['buffering_log_handler']
    buffering_log_handler.close()

    try:
        config = config_loader.yaml_loader.yaml_config_classes[tag]
    except KeyError:
        raise BadParameter(f"There is no configuration definition loaded for the tag {tag}. "
                           f"Make sure that the configuration class is imported.")

    jschema: str = json.dumps(config.model_json_schema(ref_template='/REPLACE/{model}/REPLACE/'))

    for config_tag, config_class in config_loader.yaml_loader.yaml_config_classes.items():
        jschema = jschema.replace(f'"{config_class.__name__}": {{"title": "{config_class.__name__}"',
                                  f'"{config_tag}": {{"title": "{config_tag}"')
        jschema = jschema.replace(f'"title": "{config_class.__name__}"', f'"title": "{config_tag}"')
        jschema = jschema.replace(f'/REPLACE/{config_class.__name__}/REPLACE/', f'#/definitions/{config_tag}')

    # Replace definitions of models which are not configurations
    jschema = re.sub(r'/REPLACE/(?P<name>.*?)/REPLACE/', r'#/definitions/\g<name>', jschema)

    schema = json.loads(jschema)
    title = schema['title'] if 'description' not in schema else f"{schema['title']}\n{schema['description']}\n"
    print(title)
    print(f"\nproperties: {json.dumps(schema['properties'], indent=2)}")
    if definitions:
        print(f"\n\ndefinitions: {json.dumps(schema['definitions'], indent=2)}")


if __name__ == '__main__':
    cli()
