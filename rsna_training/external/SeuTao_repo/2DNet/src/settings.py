import os


def _ensure_trailing_slash(path: str) -> str:
    if not path:
        return path
    return path if path.endswith(os.sep) else path + os.sep


train_png_dir = _ensure_trailing_slash(os.environ.get('RSNA_TRAIN_PNG_DIR', ''))
test_png_dir = _ensure_trailing_slash(os.environ.get('RSNA_TEST_PNG_DIR', ''))
concat_train_dir = _ensure_trailing_slash(os.environ.get('RSNA_CONCAT_TRAIN_DIR', ''))
concat_test_dir = _ensure_trailing_slash(os.environ.get('RSNA_CONCAT_TEST_DIR', ''))