import os,sys,warnings
import argparse
import subprocess

# Try to import pkg_resources, install setuptools if missing
try:
    import pkg_resources
except ImportError:
    print("[INFO] pkg_resources not found. Installing setuptools...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools"], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        import pkg_resources
        print("[SUCCESS] setuptools installed successfully")
    except Exception as e:
        print(f"[WARNING] Could not install setuptools: {e}")
        print("[WARNING] Package checking functionality may be limited")
        # Create a dummy pkg_resources-like object to prevent errors
        class DummyDistribution:
            def __init__(self, version="unknown"):
                self.version = version
        
        class DummyWorkingSet:
            def __iter__(self):
                return iter([])
            
            def __getitem__(self, key):
                return DummyDistribution()
        
        class DummyVersion:
            def __init__(self, version_str):
                self.version_str = version_str
            def __lt__(self, other):
                return False
            def __eq__(self, other):
                return True
        
        class DummyPkgResources:
            working_set = DummyWorkingSet()
            
            @staticmethod
            def get_distribution(pkg):
                return DummyDistribution()
            
            @staticmethod
            def parse_version(version):
                return DummyVersion(str(version))
        
        pkg_resources = DummyPkgResources()

from pathlib import Path
import pandas as pd

warnings.filterwarnings("ignore")
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Config:
    """Singleton configuration class for the PPI binding affinity project"""
    _instance = None
    _initialized = False
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config,cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_paths()
            self._setup_packages()
            self._setup_arguments()
            self.generate_tables()  # Generate tables from P2P.csv
            self.compile_cython_extensions()  # Compile Cython extensions if available
            Config._initialized = True
    
    def _setup_paths(self):
        """Setup all path configurations"""
        # Project directory structure
        self.DIR_PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

        # Data directory structure
        self.DIR_DATA = os.path.join(self.DIR_PROJECT, "data")
        self.DIR_RESULTS = os.path.join(self.DIR_PROJECT, "results")

        # Main data file
        self.FILE_P2P_CSV = os.path.join(self.DIR_DATA, "P2P.csv")

        # Dataset directories for downloaded data
        self.DIR_V2020 = os.path.join(self.DIR_DATA, "V2020")
        self.DIR_WT = os.path.join(self.DIR_DATA, "WT")
        self.DIR_MT = os.path.join(self.DIR_DATA, "MT")

        # Dataset configurations
        self.DATASET_CONFIG = {
            'V2020':{
                'name': 'PDBbind_v2020_PP',
                # short suffix used for feature file names and directories
                'suffix': 'v2020',
            },
            'WT':{
                'name': 'SKEMPI_v2_WT',
                'suffix': 'wt',
            },
            'MT':{
                'name': 'SKEMPI_v2_MT',
                'suffix': 'mt',
            },

        }
    def _setup_packages(self):
        """Setup package management configurations"""
        # Required packages and their versions
        self.REQUIRED_PACKAGES = {
            'fair-esm': '2.0.0',      # ESM protein language model
            'torch': '2.0.0',         # PyTorch for deep learning
            'numpy': '1.21.0',        # Numerical computing
            'pandas': '1.3.0',        # Data manipulation
            'biopython': '1.79',      # Biological sequence handling
            'requests': '2.26.0',     # For downloading PDB files
            'tqdm': '4.62.0',         # Progress bars
            'gudhi': '3.7.1',         # Topological data analysis library
            'scikit-learn': '1.0.2',  # Machine learning library
            'matplotlib': '3.4.3',    # Plotting library
            'seaborn': '0.11.2',      # Statistical data visualization
            'scipy': '1.7.1',         # Scientific computing
            'cython': '0.29.0',       # Cython for C++ extensions
            'pdb2pqr': '3.6.1',       # PDB2PQR tool (pip package)
        }
    
    def _setup_arguments(self):
        """Setup argument parser configurations"""
        pass

    def compile_cython_extensions(self):
        """Compile Cython extensions if available for protein analysis"""
        try:
            print("\n"+"="*60)
            print("COMPILING CYTHON EXTENSIONS...")
            print("="*60)

            # Check if Cython is available
            try:
                from Cython.Distutils import build_ext
                from Cython.Build import cythonize
                import numpy as np
            except ImportError as e:
                print(f"[ERROR] Required packages not found: {str(e)}")
                print("Please install Cython and numpy first:")
                print("pip install cython numpy")
                return False
        # Define paths for pyprotein compilation
            pyprotein_dir = os.path.join(self.DIR_PROJECT, "src", "p2p_bio", "p2p_protein")
            setup_script = os.path.join(self.DIR_PROJECT, "src", "p2p_bio", "setup_p2p_protein.py")
            
            # Check if pyprotein directory exists
            if not os.path.exists(pyprotein_dir):
                print(f"[WARNING] pyprotein directory not found: {pyprotein_dir}")
                print("Cython compilation will be skipped.")
                return False
            
            # Check if setup script exists
            if not os.path.exists(setup_script):
                print(f"[WARNING] setup_pyprotein.py not found: {setup_script}")
                print("Cython compilation will be skipped.")
                return False
            
            print(f"Compiling Cython extensions from: {pyprotein_dir}")
            
            # Change to the p2p_bio directory for compilation
            original_cwd = os.getcwd()
            p2p_bio_dir = os.path.join(self.DIR_PROJECT, "src", "p2p_bio")
            
            try:
                os.chdir(p2p_bio_dir)
                
                # Run the setup script to compile
                cmd = [sys.executable, "setup_p2p_protein.py", "build_ext", "-i"]
                print(f"Running: {' '.join(cmd)}")
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                
                if result.returncode == 0:
                    print("[SUCCESS] Cython extensions compiled successfully!")
                    if result.stdout:
                        print("Compilation output:")
                        print(result.stdout)
                    return True
                else:
                    print("[ERROR] Cython compilation failed!")
                    print("Error output:")
                    print(result.stderr)
                    return False
                    
            except subprocess.TimeoutExpired:
                print("[ERROR] Cython compilation timed out after 5 minutes")
                return False
            except Exception as e:
                print(f"[ERROR] Unexpected error during compilation: {str(e)}")
                return False
            finally:
                # Restore original working directory
                os.chdir(original_cwd)
                
        except Exception as e:
            print(f"[ERROR] Failed to compile Cython extensions: {str(e)}")
            return False
    
    def generate_tables(self):

        """Generate tables for P2P.csv and update dataset configuration"""

        try:
            # Read the main P2P.csv file
            DATASET_P2P = pd.read_csv(self.FILE_P2P_CSV)

            # Generate dataset-specific tables
            # Note: suffix values in CSV are 'V2020', 'wt', 'mt' (lowercase for wt/mt)
            V2020_TABLE = DATASET_P2P[DATASET_P2P['suffix'] == 'V2020']
            WT_TABLE = DATASET_P2P[DATASET_P2P['suffix'] == 'wt']
            MT_TABLE = DATASET_P2P[DATASET_P2P['suffix'] == 'mt']

            # Debug: Print number of rows found for each dataset
            print(f"\n[INFO] Dataset filtering results:")
            print(f"  V2020: {len(V2020_TABLE)} rows")
            print(f"  WT: {len(WT_TABLE)} rows")
            print(f"  MT: {len(MT_TABLE)} rows")

            # Store partner1 and partner2 columns as lists for each dataset
            self.PDBBIND_V2020_PP_partner1 = V2020_TABLE['partnerA'].tolist()
            self.PDBBIND_V2020_PP_partner2 = V2020_TABLE['partnerB'].tolist()
            self.SKEMPI_V2_WT_partner1 = WT_TABLE['partnerA'].tolist()
            self.SKEMPI_V2_WT_partner2 = WT_TABLE['partnerB'].tolist()
            self.SKEMPI_V2_MT_partner1 = MT_TABLE['partnerA'].tolist()
            self.SKEMPI_V2_MT_partner2 = MT_TABLE['partnerB'].tolist()

            # Update dataset configuration with tables and directories
            self.DATASET_CONFIG['V2020'].update({
                'table': V2020_TABLE,
                'dir': self.DIR_V2020
            })
            self.DATASET_CONFIG['WT'].update({
                'table': WT_TABLE,
                'dir': self.DIR_WT
            })
            self.DATASET_CONFIG['MT'].update({
                'table': MT_TABLE,
                'dir': self.DIR_MT
            })

            # Store the main table as well
            self.TABLE_P2P = DATASET_P2P

            print("[SUCCESS] Tables generated successfully from P2P.csv!")
            return True
        
        except FileNotFoundError:
            print(f'Warning: P2P.csv not found at {self.FILE_P2P_CSV}. Please ensure the file exists.'
                  )
            print("Tables will be generated when data is available.")
            return False
    
    def print_directory_structure(self):
        """Print directory structure for verification"""
        print("\nDirectory Structure:")
        print(f"DIR_PROJECT: {self.DIR_PROJECT}")
        print(f"DIR_DATA: {self.DIR_DATA}")
        print(f"DIR_RESULTS: {self.DIR_RESULTS}")
        print(f"FILE_P2P_CSV: {self.FILE_P2P_CSV}")
        print(f"DIR_V2020: {self.DIR_V2020}")
        print(f"DIR_WT: {self.DIR_WT}")
        print(f"DIR_MT: {self.DIR_MT}")

    def check_and_install_packages(self, auto_create_env=False):
        """Check if required packages are installed, install if missing
        
        Args:
            auto_create_env: If True, automatically create environment without prompting
        
        Returns:
            bool: True if packages are installed/ready, False if user needs to activate environment
        """
        print("\n" + "="*60)
        print("CHECKING PYTHON PACKAGES...")
        print("="*60)
        
        # Check for virtual environment first (always check, not just when packages are missing)
        if not self._is_in_virtual_environment():
            env_active = self._check_or_create_environment(auto_create=auto_create_env)
            if not env_active:
                # User needs to activate environment first
                print("\n" + "="*60)
                print("[ACTION REQUIRED] Please activate the virtual environment and re-run this script.")
                print("="*60)
                return False
        
        print("\nChecking required packages...")
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        
        missing_packages = []
        outdated_packages = []

        for package, version in self.REQUIRED_PACKAGES.items():
            if package not in installed_packages:
                missing_packages.append((package, version))
            else:
                # Check if version is subfficient
                installed_version = pkg_resources.get_distribution(package).version
                if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(version):
                    outdated_packages.append((package, version, installed_version))
        
        # Now install/upgrade packages
        for package, version in self.REQUIRED_PACKAGES.items():
            try:
                # Check if package is installed
                if package not in installed_packages:
                    print(f"Installing {package}...")
                    subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}>={version}"])
                    print(f"Successfully installed {package}")
                else:
                    # Check if version is sufficient
                    installed_version = pkg_resources.get_distribution(package).version
                    if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(version):
                        print(f"Upgrading {package} from {installed_version} to {version}...")
                        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", f"{package}>={version}"])
                        print(f"Successfully upgraded {package}")
                    else:
                        print(f"[SUCCESS] {package} {installed_version} is already installed")
            except Exception as e:
                print(f"Error installing {package}: {str(e)}")
                raise

        print("\nAll required packages are installed and up to date!")
        return True
    
    def check_or_create_environment(self, auto_create=False):
        """Public method to check or create virtual environment in project root
        
        Args:
            auto_create: If True, automatically create environment without prompting
        
        Returns:
            bool: True if running in venv or venv is active, False if user needs to activate
        """
        return self._check_or_create_environment(auto_create=auto_create)
    
    def _is_in_virtual_environment(self):
        """Check if the script is running in a virtual environment"""
        return hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    
    def _check_or_create_environment(self, auto_create=False):
        """Check for virtual environment and optionally create one in project root"""
        if self._is_in_virtual_environment():
            print("[OK] Running in a virtual environment")
            return True
        
        print("\n" + "="*60)
        print("[WARNING] You are not in a virtual environment!")
        print("It's recommended to create a virtual environment before installing packages.")
        print("="*60)
        
        # Check if environment already exists in project root
        env_name = "env"
        env_path = os.path.join(self.DIR_PROJECT, env_name)
        env_exists = os.path.exists(env_path) and os.path.isdir(env_path)
        
        if env_exists:
            print(f"\n[INFO] Virtual environment found at: {env_path}")
            print("However, you are not currently using it.")
            print("\nTo activate the environment, run:")
            if os.name == 'nt':  # Windows
                print(f"  {env_path}\\Scripts\\activate")
            else:  # Unix/Linux/macOS
                print(f"  source {env_path}/bin/activate")
            print("\nAfter activation, re-run this script to install packages.")
            return False
        
        # Create environment if auto_create is True, otherwise prompt
        if auto_create:
            create_env = True
        else:
            while True:
                response = input("\nWould you like to create a virtual environment? (y/n): ").lower().strip()
                if response in ['y', 'yes']:
                    create_env = True
                    break
                elif response in ['n', 'no']:
                    create_env = False
                    break
                else:
                    print("Please enter 'y' or 'n'.")
        
        if create_env:
            user_env_name = input(f"Enter environment name (default: {env_name}): ").strip()
            if user_env_name:
                env_name = user_env_name
                env_path = os.path.join(self.DIR_PROJECT, env_name)
            
            print(f"\nCreating virtual environment '{env_name}' in project directory...")
            try:
                # Create virtual environment in project root
                subprocess.check_call([sys.executable, "-m", "venv", env_path])
                print(f"[SUCCESS] Virtual environment '{env_name}' created successfully at: {env_path}")

                # Install basic packages in the new environment
                print(f"Installing basic packages in the new environment...")
                if os.name == 'nt':  # Windows
                    pip_path = os.path.join(env_path, "Scripts", "pip")
                    python_path = os.path.join(env_path, "Scripts", "python")
                else:  # Unix/Linux/macOS
                    pip_path = os.path.join(env_path, "bin", "pip")
                    python_path = os.path.join(env_path, "bin", "python")
                
                # Upgrade pip first
                subprocess.check_call([pip_path, "install", "--upgrade", "pip"])
                print("[SUCCESS] pip upgraded")
                
                # Install pandas (required for config to work)
                subprocess.check_call([pip_path, "install", "pandas>=1.3.0"])
                print("[SUCCESS] pandas installed")
                
                # Provide activation instructions
                print(f"\n" + "="*60)
                print("ENVIRONMENT CREATED SUCCESSFULLY!")
                print("="*60)
                print(f"\nTo activate the environment, run:")
                if os.name == 'nt':  # Windows
                    print(f"  {env_path}\\Scripts\\activate")
                else:  # Unix/Linux/macOS
                    print(f"  source {env_path}/bin/activate")
                print(f"\nOr use the python directly:")
                print(f"  {python_path}")
                print(f"\nAfter activation, re-run this script to install remaining packages.")
                print("="*60)
                
                return False  # Return False because user needs to activate
                    
            except Exception as e:
                print(f"[ERROR] Error creating virtual environment: {str(e)}")
                print("Please create the environment manually and try again.")
                return False
        else:
            print("\nProceeding without virtual environment...")
            print("Note: This may install packages globally, which could cause conflicts.")
            return False
    
    def _prompt_create_environment(self):
        """Prompt user to create a virtual environment (deprecated, use _check_or_create_environment)"""
        return self._check_or_create_environment(auto_create=False)
    
    def check_external_tools(self):
        """Check if required external tools are installed and accessible"""
        print("\n"+"="*60)
        print("CHECKING EXTERNAL TOOLS...")
        print("="*60)

        # Get project root and bin directory
        bin_dir = os.path.join(self.DIR_PROJECT, "bin")

        # Create bin directory if it doesn't exist
        if not os.path.exists(bin_dir):
            print(f"[INFO] Creating bin directory: {bin_dir}")
            os.makedirs(bin_dir, exist_ok=True)
     
        # Check profix and scap (part of JACKAL)
        print("\n1. Checking for JACKAL tools (profix, scap)...")
        profix_path = os.path.join(bin_dir, "profix")
        scap_path = os.path.join(bin_dir, "scap")

        # Check if tools exist in bin directory (local installation)
        profix_exists = os.path.exists(profix_path) and os.access(profix_path, os.X_OK)
        scap_exists = os.path.exists(scap_path) and os.access(scap_path, os.X_OK)

        # Check if tools can be run globally from PATH
        profix_runnable_global = False
        scap_runnable_global = False
        
        # Try to make local tools executable
        if profix_exists:
            try:
                subprocess.run(['chmod', '+x', profix_path], check=True)
                print(f"[SUCCESS] Made local profix executable: {profix_path}")
            except Exception as e:
                print(f"Warning: Could not chmod local profix: {e}")

        if scap_exists:
            try:
                subprocess.run(['chmod', '+x', scap_path], check=True)
                print(f"[SUCCESS] Made local scap executable: {scap_path}")
            except Exception as e:
                print(f"Warning: Could not chmod local scap: {e}")

        # Test local tools
        profix_runnable_local = False
        scap_runnable_local = False
        
        if profix_exists:
            try:
                result = subprocess.run([profix_path, '--help'],
                                        capture_output=True, text=True, timeout=5)
                profix_runnable_local = (result.returncode == 0)
                if profix_runnable_local:
                    print(f"[SUCCESS] Local profix is runnable: {profix_path}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        if scap_exists:
            try:
                result = subprocess.run([scap_path, '--help'],
                                        capture_output=True, text=True, timeout=5)
                scap_runnable_local = (result.returncode == 0)
                if scap_runnable_local:
                    print(f"[SUCCESS] Local scap is runnable: {scap_path}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        # Test global tools
        try:
            result = subprocess.run(['profix', '--help'],
                                    capture_output=True, text=True, timeout=5)
            profix_runnable_global = (result.returncode == 0)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        try:
            result = subprocess.run(['scap', '--help'],
                                    capture_output=True, text=True, timeout=5)
            scap_runnable_global = (result.returncode == 0)
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Summary
        profix_available = profix_runnable_local or profix_runnable_global
        scap_available = scap_runnable_local or scap_runnable_global
        
        if profix_available and scap_available:
            print("[SUCCESS] JACKAL tools (profix, scap) are available and runnable.")
            if profix_runnable_local:
                print(f"  Using local profix: {profix_path}")
            elif profix_runnable_global:
                print("  Using global profix command")
            if scap_runnable_local:
                print(f"  Using local scap: {scap_path}")
            elif scap_runnable_global:
                print("  Using global scap command")
        else:
            print("[WARNING] profix and/or scap are not available")
            print(f"Status:")
            print(f"  profix - local: {profix_runnable_local} (exists: {profix_exists})")
            print(f"  profix - global: {profix_runnable_global}")
            print(f"  scap - local: {scap_runnable_local} (exists: {scap_exists})")
            print(f"  scap - global: {scap_runnable_global}")
            print("\nInstallation instructions:")
            print("1. Download JACKAL from: http://honig.c2b2.columbia.edu/software/jackal/")
            print("2. Extract and compile JACKAL:")
            print("   tar -xzf jackal.tar.gz")
            print("   cd jackal")
            print("   make")
            print("3. Add JACKAL executables to your PATH:")
            print("   - Copy profix and scap to a directory in your PATH")
            print("   - Or add the JACKAL directory to your PATH")
            print("4. Alternative: Copy executables to the bin directory:")
            print(f"   cp profix {bin_dir}/")
            print(f"   cp scap {bin_dir}/")
            print("   chmod +x {bin_dir}/profix")
            print(f"   chmod +x {bin_dir}/scap")
            
            # Offer to help with installation
            response = input("\nWould you like help installing JACKAL? (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                self._install_jackal_guide(bin_dir)
        # Check jackal.dir file
        print("\n2. Checking jackal.dir configuration...")
        jackal_dir_file = os.path.join(bin_dir, "jackal.dir")

        if os.path.exists(jackal_dir_file):
            try:
                with open(jackal_dir_file, 'r') as f:
                    jackal_lib_path = f.read().strip()
                
                if os.path.exists(jackal_lib_path):
                    # Check if the directory contains expected JACKAL library files
                    expected_files = ['libjackal.a', 'libjackal.so', 'jackal.lib']
                    found_files = []
                    
                    if os.path.isdir(jackal_lib_path):
                        # Check for library files in the directory
                        for file in os.listdir(jackal_lib_path):
                            if file in expected_files:
                                found_files.append(file)
                    else:
                        # If it's a file, check if it's one of the expected library files
                        if os.path.basename(jackal_lib_path) in expected_files:
                            found_files.append(os.path.basename(jackal_lib_path))
                    
                    if found_files:
                        print(f"[SUCCESS] jackal.dir points to valid JACKAL library: {jackal_lib_path}")
                        print(f"  Found library files: {', '.join(found_files)}")
                    else:
                        print(f"[WARNING] jackal.dir points to existing path but no JACKAL library files found: {jackal_lib_path}")
                        print(f"  Expected files: {', '.join(expected_files)}")
                        print("  Please update jackal.dir to point to the correct JACKAL library location")
                        print("  The directory should contain libjackal.a, libjackal.so, or jackal.lib")
                else:
                    print(f"[WARNING] jackal.dir points to non-existent path: {jackal_lib_path}")
                    print("Please update jackal.dir to point to the correct JACKAL library location")
            except Exception as e:
                print(f"[ERROR] Could not read jackal.dir: {str(e)}")
        else:
            print("[WARNING] jackal.dir file not found")
            print(f"Expected location: {jackal_dir_file}")
            print("Please create jackal.dir file with the path to your JACKAL library")
        
        # Check pdb2pqr
        print("\n3. Checking pdb2pqr...")
        try:
            result = subprocess.run(['pdb2pqr', '--version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("[SUCCESS] pdb2pqr is available")
                print(f"Version info: {result.stdout.strip()}")
            else:
                print("[WARNING] pdb2pqr is installed but may have issues")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("[WARNING] pdb2pqr is not found in PATH")
            print("Installation instructions:")
            print("1. Download from: http://www.poissonboltzmann.org/pdb2pqr/")
            print("2. Or install via conda: conda install -c conda-forge pdb2pqr")
            print("3. Or install via pip: pip install pdb2pqr")

        # Check mibpb
        print("\n4. Checking mibpb...")
        try:
            result = subprocess.run(['mibpb', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("[SUCCESS] mibpb is available")
            else:
                print("[WARNING] mibpb is installed but may have issues")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("[WARNING] mibpb is not found in PATH")
            print("Installation instructions:")
            print("1. Visit: https://weilab.math.msu.edu/MIBPB/")
            print("2. For academic/governmental users: Register and download for free")
            print("3. For industrial/commercial users: Contact wei@math.msu.edu for licensing")
            print("4. For specific compiler versions: Contact wei@math.msu.edu for binaries")
            print("5. Follow the installation instructions provided with the download")
            print("6. Add mibpb to your PATH or place it in the bin directory")
        
        # Check ms_intersection
        print("\n5. Checking ms_intersection...")
        try:
            result = subprocess.run(['ms_intersection', '--help'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                print("[SUCCESS] ms_intersection is available")
            else:
                print("[WARNING] ms_intersection is installed but may have issues")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("[WARNING] ms_intersection is not found in PATH")
            print("Installation instructions:")
            print("1. Visit: https://weilab.math.msu.edu/ESES/")
            print("2. ESES is open source and available on GitHub (Weilab)")
            print("3. ms_intersection is part of the ESES software package")
            print("4. Follow the installation instructions from the ESES repository")
            print("5. Add ms_intersection to your PATH or place it in the bin directory")
        
        # Note about optional tools
        print("\n" + "="*60)
        print("NOTE: MIBPB and MS_INTERSECTION are OPTIONAL")
        print("="*60)
        print("These tools are only required if you plan to use electrostatics features.")
        print("If you don't need electrostatics calculations, you can skip installing these tools.")
        print("The core protein-protein interaction binding affinity prediction will still work.")
        print("="*60)
        
        print("\n" + "="*60)
        print("EXTERNAL TOOLS CHECK COMPLETED")
        print("="*60)
        
        # Return summary
        tools_status = {
            'profix': profix_available,  # Local or global availability
            'scap': scap_available,      # Local or global availability
            'profix_local': profix_runnable_local,
            'profix_global': profix_runnable_global,
            'scap_local': scap_runnable_local,
            'scap_global': scap_runnable_global,
            'jackal_dir': os.path.exists(jackal_dir_file),
            'pdb2pqr': self._check_tool_in_path('pdb2pqr'),
            'mibpb': self._check_tool_in_path('mibpb'),
            'ms_intersection': self._check_tool_in_path('ms_intersection')
        }
        
        return tools_status
    def _check_tool_in_path(self, tool_name):
        """Helper method to check if a tool is available in PATH"""
        try:
            result = subprocess.run([tool_name, '--help'], 
                                  capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _install_jackal_guide(self, bin_dir):
        """Interactive guide to install JACKAL"""
        print("\n" + "="*60)
        print("JACKAL INSTALLATION GUIDE")
        print("="*60)
        
        print("1. Download JACKAL:")
        print("   - Visit: http://honig.c2b2.columbia.edu/software/jackal/")
        print("   - Download the latest version")
        
        print("\n2. Extract and compile:")
        print("   tar -xzf jackal.tar.gz")
        print("   cd jackal")
        print("   make")
        
        print("\n3. Copy executables:")
        print(f"   cp profix {bin_dir}/")
        print(f"   cp scap {bin_dir}/")
        
        print("\n4. Set permissions:")
        print(f"   chmod +x {bin_dir}/profix")
        print(f"   chmod +x {bin_dir}/scap")
        
        print("\n5. Test installation:")

        print("   profix --help")
        print("   scap --help")
 
        print("\n6. Create jackal.dir file:")
        jackal_lib_path = input("Enter the path to your JACKAL library directory: ").strip()
        if jackal_lib_path:
            jackal_dir_file = os.path.join(bin_dir, "jackal.dir")
            try:
                with open(jackal_dir_file, 'w') as f:
                    f.write(jackal_lib_path)
                print(f"[SUCCESS] Created jackal.dir file: {jackal_dir_file}")
            except Exception as e:
                print(f"[ERROR] Could not create jackal.dir file: {str(e)}")
        
        print("\nAfter installation, re-run this check to verify everything is working.")
        print("="*60)

    def create_main_parser(self):
        """Create argument parser for main program control"""
        parser = argparse.ArgumentParser(description='Protein-Protein Interaction Binding Affinity Prediction')
        
        # Model selection
        parser.add_argument('--model', type=str, 
                          choices=['esm', 'bio', 'hybrid'],
                          default='hybrid',
                          help='Model type to use for prediction')
        
        # Training parameters
        parser.add_argument('--batch_size', type=int, default=32,
                          help='Batch size for training')
        parser.add_argument('--epochs', type=int, default=100,
                          help='Number of training epochs')
        parser.add_argument('--learning_rate', type=float, default=0.001,
                          help='Learning rate for training')
        
        # Data parameters
        parser.add_argument('--train_ratio', type=float, default=0.8,
                          help='Ratio of data to use for training')
        parser.add_argument('--val_ratio', type=float, default=0.1,
                          help='Ratio of data to use for validation')
        parser.add_argument('--test_ratio', type=float, default=0.1,
                          help='Ratio of data to use for testing')
        
        # Feature extraction
        parser.add_argument('--use_esm', action='store_true',
                          help='Use ESM features')
        parser.add_argument('--use_sequence', action='store_true',
                          help='Use sequence features')
        parser.add_argument('--use_structure', action='store_true',
                          help='Use structure features')
        
        # Output and logging
        parser.add_argument('--output_dir', type=str, default='results',
                          help='Directory for output files')
        parser.add_argument('--log_level', type=str, 
                          choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                          default='INFO',
                          help='Logging level')
        parser.add_argument('--save_model', action='store_true',
                          help='Save trained model')
        
        # Hardware
        parser.add_argument('--device', type=str, 
                          choices=['cpu', 'cuda', 'auto'],
                          default='auto',
                          help='Device to use for computation')
        parser.add_argument('--num_workers', type=int, default=4,
                          help='Number of workers for data loading')
        
        return parser
    
    def parse_args(self):
        """Parse command line arguments for main program"""
        main_parser = self.create_main_parser()
        return main_parser.parse_args()

    def get_setup_status(self):
        """Generate a comprehensive status report of what the user needs to set up"""
        print("\n" + "="*80)
        print("PROTEIN-PROTEIN INTERACTION BINDING AFFINITY PREDICTION - SETUP STATUS")
        print("="*80)
        
        # Check project structure
        print("\n[DIRECTORIES] PROJECT STRUCTURE:")
        project_ok = True
        if not os.path.exists(self.DIR_PROJECT):
            print(f"[ERROR] Project directory not found: {self.DIR_PROJECT}")
            project_ok = False
        else:
            print(f"[OK] Project directory: {self.DIR_PROJECT}")
        
        if not os.path.exists(self.DIR_DATA):
            print(f"[ERROR] Data directory not found: {self.DIR_DATA}")
            print(f"   -> Create: mkdir -p {self.DIR_DATA}")
            project_ok = False
        else:
            print(f"[OK] Data directory: {self.DIR_DATA}")
        
        if not os.path.exists(self.DIR_RESULTS):
            print(f"[ERROR] Results directory not found: {self.DIR_RESULTS}")
            print(f"   -> Create: mkdir -p {self.DIR_RESULTS}")
            project_ok = False
        else:
            print(f"[OK] Results directory: {self.DIR_RESULTS}")
        
        # Check main data file
        print(f"\n[DATA] DATA FILES:")
        if not os.path.exists(self.FILE_P2P_CSV):
            print(f"[ERROR] Main data file not found: {self.FILE_P2P_CSV}")
            print(f"   -> This file should contain protein-protein interaction data")
            print(f"   -> Expected columns: partner1, partner2, source, binding_affinity")
            project_ok = False
        else:
            print(f"[OK] Main data file: {self.FILE_P2P_CSV}")
            try:
                df = pd.read_csv(self.FILE_P2P_CSV)
                print(f"   -> Contains {len(df)} protein pairs")
                print(f"   -> Sources: {df['source'].unique().tolist()}")
            except Exception as e:
                print(f"   [WARNING] Error reading file: {str(e)}")
        
        # Check dataset directories
        print(f"\n[DATASETS] DATASET DIRECTORIES:")
        # Use the path attributes defined in _setup_paths (DIR_V2020, DIR_WT, DIR_MT)
        datasets = [
            ('PDBbind_v2020', self.DIR_V2020),
            ('SKEMPI_v2_WT', self.DIR_WT),
            ('SKEMPI_v2_MT', self.DIR_MT)
        ]
        
        for name, path in datasets:
            if not os.path.exists(path):
                print(f"[ERROR] {name} directory not found: {path}")
                print(f"   -> Create: mkdir -p {path}")
                project_ok = False
            else:
                file_count = len([f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))])
                print(f"[OK] {name} directory: {path} ({file_count} files)")
        
        # Check Python packages
        print(f"\n[PYTHON] PYTHON PACKAGES:")
        installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
        
        missing_packages = []
        outdated_packages = []
        critical_packages = ['torch', 'fair-esm', 'numpy', 'pandas', 'biopython']
        
        for package, version in self.REQUIRED_PACKAGES.items():
            if package not in installed_packages:
                missing_packages.append((package, version))
                if package in critical_packages:
                    print(f"[ERROR] CRITICAL: {package} {version} not installed")
                else:
                    print(f"[ERROR] {package} {version} not installed")
            else:
                installed_version = pkg_resources.get_distribution(package).version
                if pkg_resources.parse_version(installed_version) < pkg_resources.parse_version(version):
                    outdated_packages.append((package, version, installed_version))
                    print(f"[WARNING] {package} outdated: {installed_version} < {version}")
                else:
                    print(f"[OK] {package} {installed_version}")
        
        # Check external tools
        print(f"\n[TOOLS] EXTERNAL TOOLS:")
        
        # Check JACKAL tools
        bin_dir = os.path.join(self.DIR_PROJECT, "bin")
        profix_path = os.path.join(bin_dir, "profix")
        scap_path = os.path.join(bin_dir, "scap")
        
        profix_ok = os.path.exists(profix_path) and os.access(profix_path, os.X_OK)
        scap_ok = os.path.exists(scap_path) and os.access(scap_path, os.X_OK)
        
        if profix_ok:
            print(f"[OK] profix (local): {profix_path}")
        else:
            print(f"[ERROR] profix not found in: {profix_path}")
        
        if scap_ok:
            print(f"[OK] scap (local): {scap_path}")
        else:
            print(f"[ERROR] scap not found in: {scap_path}")
        
        # Check global tools
        try:
            subprocess.run(['profix', '--help'], capture_output=True, timeout=5)
            print(f"[OK] profix (global)")
        except:
            if not profix_ok:
                print(f"[ERROR] profix not available globally")
        
        try:
            subprocess.run(['scap', '--help'], capture_output=True, timeout=5)
            print(f"[OK] scap (global)")
        except:
            if not scap_ok:
                print(f"[ERROR] scap not available globally")
        
        # Check jackal.dir
        jackal_dir_file = os.path.join(bin_dir, "jackal.dir")
        if os.path.exists(jackal_dir_file):
            try:
                with open(jackal_dir_file, 'r') as f:
                    jackal_lib_path = f.read().strip()
                if os.path.exists(jackal_lib_path):
                    print(f"[OK] jackal.dir: {jackal_lib_path}")
                else:
                    print(f"[ERROR] jackal.dir points to non-existent path: {jackal_lib_path}")
            except:
                print(f"[ERROR] Could not read jackal.dir file")
        else:
            print(f"[ERROR] jackal.dir file not found: {jackal_dir_file}")
        
        # Check optional tools
        optional_tools = ['pdb2pqr', 'mibpb', 'ms_intersection']
        for tool in optional_tools:
            try:
                subprocess.run([tool, '--help'], capture_output=True, timeout=5)
                print(f"[OK] {tool} (optional)")
            except:
                print(f"[WARNING] {tool} not found (optional)")
        
        # Check Cython extensions
        print(f"\n[CYTHON] CYTHON EXTENSIONS:")
        pyprotein_dir = os.path.join(self.DIR_PROJECT, "src", "p2p_bio", "p2p_protein")
        if os.path.exists(pyprotein_dir):
            print(f"[OK] pyprotein directory: {pyprotein_dir}")
            # Check for compiled extensions
            compiled_files = []
            for ext in ['.so', '.pyd', '.dll']:
                compiled_files.extend([f for f in os.listdir(pyprotein_dir) if f.endswith(ext)])
            if compiled_files:
                print(f"[OK] Compiled extensions found: {', '.join(compiled_files)}")
            else:
                print(f"[WARNING] No compiled extensions found - will compile on first run")
        else:
            print(f"[ERROR] pyprotein directory not found: {pyprotein_dir}")
        
        # Summary and recommendations
        print(f"\n" + "="*80)
        print("SETUP SUMMARY & RECOMMENDATIONS")
        print("="*80)
        
        if not project_ok:
            print(f"\n[ERROR] CRITICAL ISSUES FOUND:")
            print(f"   -> Fix project structure issues above")
            print(f"   -> Ensure P2P.csv data file is available")
        
        if missing_packages:
            print(f"\n[PACKAGES] MISSING PACKAGES ({len(missing_packages)}):")
            for package, version in missing_packages:
                print(f"   -> pip install {package}>={version}")
            print(f"\n   Or install all at once:")
            packages_str = " ".join([f"{p}>={v}" for p, v in missing_packages])
            print(f"   -> pip install {packages_str}")
        
        if outdated_packages:
            print(f"\n[UPDATE] OUTDATED PACKAGES ({len(outdated_packages)}):")
            for package, version, current in outdated_packages:
                print(f"   -> pip install --upgrade {package}>={version}")
        
        if not (profix_ok or scap_ok):
            print(f"\n[JACKAL] JACKAL TOOLS REQUIRED:")
            print(f"   -> Download from: http://honig.c2b2.columbia.edu/software/jackal/")
            print(f"   -> Extract and compile: tar -xzf jackal.tar.gz && cd jackal && make")
            print(f"   -> Copy to bin directory:")
            print(f"     cp profix {bin_dir}/")
            print(f"     cp scap {bin_dir}/")
            print(f"     chmod +x {bin_dir}/profix")
            print(f"     chmod +x {bin_dir}/scap")
        
        if not os.path.exists(jackal_dir_file):
            print(f"\n[CONFIG] JACKAL LIBRARY CONFIGURATION:")
            print(f"   -> Create jackal.dir file: {jackal_dir_file}")
            print(f"   -> Add path to JACKAL library directory")
        
        print(f"\n[SETUP] QUICK SETUP COMMANDS:")
        print(f"   # Create directories")
        print(f"   mkdir -p {self.DIR_DATA} {self.DIR_RESULTS}")
        print(f"   mkdir -p {self.DIR_V2020} {self.DIR_WT} {self.DIR_MT}")
        print(f"   mkdir -p {bin_dir}")
        print(f"   ")
        print(f"   # Install Python packages")
        if missing_packages:
            packages_str = " ".join([f"{p}>={v}" for p, v in missing_packages])
            print(f"   pip install {packages_str}")
        print(f"   ")
        print(f"   # Run setup check")
        print(f"   python src/config.py")
        
        print(f"\n[NEXT] NEXT STEPS:")
        print(f"   1. Ensure P2P.csv data file is in {self.DIR_DATA}/")
        print(f"   2. Install missing Python packages")
        print(f"   3. Install JACKAL tools (if not already available)")
        print(f"   4. Configure jackal.dir file")
        print(f"   5. Run: python src/config.py to verify setup")
        print(f"   6. Start using the prediction models!")
        
        print(f"\n" + "="*80)
        
        return {
            'project_ok': project_ok,
            'missing_packages': missing_packages,
            'outdated_packages': outdated_packages,
            'jackal_ok': profix_ok and scap_ok,
            'jackal_dir_ok': os.path.exists(jackal_dir_file)
        }

# Create global instance
config = Config()

# Export commonly used attributes for backward compatibility
DIR_DATA = config.DIR_DATA
DIR_RESULTS = config.DIR_RESULTS
FILE_P2P_CSV = config.FILE_P2P_CSV
DIR_V2020 = config.DIR_V2020
DIR_WT = config.DIR_WT
DIR_MT = config.DIR_MT
DIR_PROJECT = config.DIR_PROJECT
DATASET_CONFIG = config.DATASET_CONFIG

# Export table-related attributes (if available)
TABLE_P2P = getattr(config, 'TABLE_P2P', None)
# TABLE_PDBBIND_V2020_PP is the V2020 dataset table (previously misnamed/mis-keyed)
TABLE_PDBBIND_V2020_PP = config.DATASET_CONFIG.get('V2020', {}).get('table', None)
TABLE_SKEMPI_V2_WT = config.DATASET_CONFIG.get('WT', {}).get('table', None)
TABLE_SKEMPI_V2_MT = config.DATASET_CONFIG.get('MT', {}).get('table', None)

# Export functions
create_main_parser = config.create_main_parser
parse_args = config.parse_args
check_and_install_packages = config.check_and_install_packages
check_or_create_environment = config.check_or_create_environment
print_directory_structure = config.print_directory_structure
generate_tables = config.generate_tables
check_external_tools = config.check_external_tools
compile_cython_extensions = config.compile_cython_extensions
get_setup_status = config.get_setup_status

# Add project directory to sys.path
sys.path.append(config.DIR_PROJECT)

# =============================================================================
# CONVENIENT IMPORT HELPER
# =============================================================================

def get_config():
    """Get the global config instance - loads only once"""
    return config

# Make it easy to import specific items
__all__ = [
    'config',
    'get_config',
    'DIR_DATA',
    'DIR_RESULTS',
    'FILE_P2P_CSV',
    'DIR_V2020',
    'DIR_WT',
    'DIR_MT',
    'DIR_PROJECT',
    'DATASET_CONFIG',
    'TABLE_P2P',
    'TABLE_PDBBIND_V2020_PP',
    'TABLE_SKEMPI_V2_WT',
    'TABLE_SKEMPI_V2_MT',
    'create_main_parser',
    'parse_args',
    'check_and_install_packages',
    'check_or_create_environment',
    'print_directory_structure',
    'generate_tables',
    'check_external_tools',
    'compile_cython_extensions',
    'get_setup_status'
]

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Example usage
    print("Configuration module loaded successfully!")
    
    # # Generate comprehensive setup status report
    # print("\n" + "="*80)
    # print("RUNNING COMPREHENSIVE SETUP CHECK...")
    # print("="*80)
    # status = config.get_setup_status()
    
    # # Additional detailed checks
    # print("\n" + "="*80)
    # print("RUNNING DETAILED CHECKS...")
    # print("="*80)
    
    # config.print_directory_structure()
    
    # Check packages (this will also check/create virtual environment)
    packages_ready = config.check_and_install_packages()
    
    if not packages_ready:
        # Environment was created but not activated, exit gracefully
        sys.exit(0)

    # Check external tools
    config.check_external_tools()
    
    # Compile Cython extensions
    config.compile_cython_extensions()
    
    # # Final summary
    # print("\n" + "="*80)
    # print("SETUP CHECK COMPLETED")
    # print("="*80)
    # print(f"Project structure: {'[OK]' if status['project_ok'] else '[ERROR] ISSUES'}")
    # missing_count = len(status['missing_packages']) if status['missing_packages'] else 0
    # print(f"Python packages: {'[OK]' if not status['missing_packages'] else f'[ERROR] {missing_count} missing'}")
    # print(f"JACKAL tools: {'[OK]' if status['jackal_ok'] else '[ERROR] MISSING'}")
    # print(f"JACKAL config: {'[OK]' if status['jackal_dir_ok'] else '[ERROR] MISSING'}")
    # print("="*80)
