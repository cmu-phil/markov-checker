import os
import platform
import requests
import shutil
import tarfile
import zipfile

# Constants for Java JDK URLs
WINDOWS_JDK_URL = "https://corretto.aws/downloads/latest/amazon-corretto-21-x64-windows-jdk.zip"
MAC_JDK_URL_ARM = "https://corretto.aws/downloads/latest/amazon-corretto-21-aarch64-macos-jdk.tar.gz"
MAC_JDK_URL_X86 = "https://corretto.aws/downloads/latest/amazon-corretto-21-x64-macos-jdk.tar.gz"
LINUX_JDK_URL = "https://corretto.aws/downloads/latest/amazon-corretto-21-x64-linux-jdk.tar.gz"
TETRAD_URL = "https://s01.oss.sonatype.org/content/repositories/releases/io/github/cmu-phil/tetrad-gui/7.6.5/tetrad-gui-7.6.5-launch.jar"
TETRAD_PATH = os.path.abspath("inst/tetrad-gui-7.6.5-launch.jar")
JAVA_DIR = os.path.abspath("inst/jdk-21.0.12.jdk")


# Function to check internet connection
def check_internet_connection():
    try:
        response = requests.get("http://www.google.com", timeout=5)
        return True if response.status_code == 200 else False
    except requests.ConnectionError:
        return False


# Function to download a file from a URL
def download_file(url, destfile):
    if not check_internet_connection():
        raise ConnectionError("No internet connection. Please check your network and try again.")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(destfile, 'wb') as file:
            shutil.copyfileobj(response.raw, file)
        print(f"Download successful: {destfile}")
    except requests.exceptions.RequestException as e:
        print(f"An error occurred during the download: {e}")
        raise e


# Function to install Java JDK locally based on the operating system and architecture
def install_local_java(java_dir=JAVA_DIR):
    print("Starting Java installation...")

    if os.path.exists(java_dir):
        print(f"Java JDK is already installed at: {java_dir}")
        return java_dir

    try:
        os.makedirs('inst', exist_ok=True)

        system_platform = platform.system()
        architecture = platform.machine()
        print(f"Detected platform: {system_platform}, Architecture: {architecture}")

        if system_platform == "Windows":
            download_file(WINDOWS_JDK_URL, "inst/jdk.zip")
            with zipfile.ZipFile("inst/jdk.zip", 'r') as zip_ref:
                zip_ref.extractall(java_dir)
            os.remove("inst/jdk.zip")

        elif system_platform == "Darwin":  # macOS
            if architecture in ["arm64", "aarch64"]:
                download_file(MAC_JDK_URL_ARM, "inst/jdk.tar.gz")
            else:
                download_file(MAC_JDK_URL_X86, "inst/jdk.tar.gz")
            with tarfile.open("inst/jdk.tar.gz", 'r:gz') as tar_ref:
                tar_ref.extractall(java_dir)
            os.remove("inst/jdk.tar.gz")

        elif system_platform == "Linux":
            download_file(LINUX_JDK_URL, "inst/jdk.tar.gz")
            with tarfile.open("inst/jdk.tar.gz", 'r:gz') as tar_ref:
                tar_ref.extractall(java_dir)
            os.remove("inst/jdk.tar.gz")

        print(f"Java JDK installed at: {java_dir}")
    except Exception as e:
        print(f"An error occurred during Java installation: {e}")
        raise e

    return java_dir


# Function to set the JAVA_HOME environment variable
def set_java_home(java_dir):
    java_home = os.path.join(java_dir, "Contents", "Home") if platform.system() == "Darwin" else java_dir
    print(f"Setting JAVA_HOME to: {java_home}")

    if not os.path.exists(java_home):
        raise FileNotFoundError(f"The specified JAVA_HOME directory does not exist: {java_home}")

    os.environ["JAVA_HOME"] = java_home
    os.environ["PATH"] = f"{os.path.join(java_home, 'bin')}:{os.environ['PATH']}"
    print(f"JAVA_HOME is set to: {os.getenv('JAVA_HOME')}")


# Function to download Tetrad
def download_tetrad():
    print("Starting Tetrad download...")

    os.makedirs("inst", exist_ok=True)

    if os.path.exists(TETRAD_PATH):
        print(f"File already exists at: {TETRAD_PATH}")
    else:
        download_file(TETRAD_URL, TETRAD_PATH)
        print(f"File downloaded successfully to: {TETRAD_PATH}")


# Main setup function to install Java and Tetrad
def setup_tetrad_environment():
    install_local_java(java_dir=JAVA_DIR)
    download_tetrad()
    set_java_home(JAVA_DIR)
    print("Tetrad environment setup complete.")


# Example usage
if __name__ == "__main__":
    setup_tetrad_environment()
