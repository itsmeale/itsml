import os


def check_import_order():
    os.system("isort --check ./itsml/ --skip __init__.py --gitignore --dont-follow-links --verbose")


def check_code_formatting():
    os.system("black --check ./itsml/ --exclude __init__.py --verbose")


def sort_import_order():
    os.system("isort ./itsml/ ./tests/ --skip __init__.py --gitignore --dont-follow-links --verbose")


def do_code_formatting():
    os.system("black ./itsml/ ./tests/ --exclude __init__.py --exclude .ipynb_checkpoints/ --verbose")


def linter():
    os.system("pylama ./itsml/ ./tests/")


def run_tests():
    os.system("pytest ./tests/ --verbose --color=yes --code-highlight=yes")
