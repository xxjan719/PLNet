# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Changed
- **Removed pandas dependency from `machine_learning.py` and `config.py`**
  - Replaced `pd.read_csv()` with Python's built-in `csv.DictReader` module
  - Replaced pandas DataFrame operations with list comprehensions and dictionaries
  - Replaced `pd.Series().str.startswith()` with Python list comprehensions using `str.startswith()`
  - Removed pandas from `REQUIRED_PACKAGES` in `config.py`
  - Removed pandas installation code from environment setup
  - Removed pandas from critical packages list
  - **Impact**: Users no longer need to install pandas for basic CSV reading operations in these files
  - **Note**: Other files (`feature_generation.py`, `utils.py`) still use pandas for advanced DataFrame operations

### Documentation
- **Enhanced README.md installation instructions**
  - Added detailed step-by-step installation guide based on `config.py` functionality
  - Documented all Python packages that are automatically installed
  - Added clear instructions for virtual environment setup
  - Explained what `config.py` does at each step
  - Added platform-specific activation commands (Linux/macOS/Windows)
  - Clarified the two-step installation process (create env, then install packages)

### Files Modified
- `src/machine_learning.py`: Removed pandas dependency, uses `csv` module
- `src/config.py`: Removed pandas dependency, uses `csv` module, stores tables as lists of dicts
- `README.md`: Enhanced installation documentation

## [Previous Versions]

*Note: Previous changelog entries would be added here as the project evolves*

