from pathlib import Path
import cars_fast

def test_runtime_paths_are_inside_this_checkout():
    for path in (cars_fast.ORIGINAL_RUN, cars_fast.DATA, cars_fast.FAST_RUNS):
        assert path.resolve().is_relative_to(cars_fast.ROOT.resolve())
    assert cars_fast._model_config_path().exists()

def test_runtime_package_was_preserved():
    import hashlib
    digest = hashlib.sha256()
    for path in sorted((cars_fast.ROOT/'cars_r').glob('*.py')):
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    assert digest.hexdigest() == '1c6eb29e6e3dd84a4e1d2f30ec560d04ced855084237dab9526239df36ab1914'
