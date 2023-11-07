Usage
=====

Installation
------------

Install Anaconda environment manager
------------------------------------


Prerequisites
-------------

To use GUI packages with Linux, you will need to install the following extended dependencies for Qt:

Debian

.. code-block:: console

    apt-get install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

RedHat

.. code-block:: console

    yum install libXcomposite libXcursor libXi libXtst libXrandr alsa-lib mesa-libEGL libXdamage mesa-libGL libXScrnSaver

ArchLinux

.. code-block:: console

    pacman -Sy libxau libxi libxss libxtst libxcursor libxcomposite libxdamage libxfixes libxrandr libxrender mesa-libgl  alsa-lib libglvnd

OpenSuse/SLES

.. code-block:: console

    zypper install libXcomposite1 libXi6 libXext6 libXau6 libX11-6 libXrandr2 libXrender1 libXss1 libXtst6 libXdamage1 libXcursor1 libxcb1 libasound2  libX11-xcb1 Mesa-libGL1 Mesa-libEGL1

Gentoo

.. code-block:: console

    emerge x11-libs/libXau x11-libs/libxcb x11-libs/libX11 x11-libs/libXext x11-libs/libXfixes x11-libs/libXrender x11-libs/libXi x11-libs/libXcomposite x11-libs/libXrandr x11-libs/libXcursor x11-libs/libXdamage x11-libs/libXScrnSaver x11-libs/libXtst media-libs/alsa-lib media-libs/mesa


Installation
------------

For x86 systems.

In your browser, download the Anaconda installer for Linux.

Search for “terminal” in your applications and click to open.

(Recommended) Verify the installer’s data integrity with SHA-256. For more information on hash verification, see cryptographic hash validation.

In the terminal, run the following:

.. code-block:: console

    shasum -a 256 /PATH/FILENAME
    # Replace /PATH/FILENAME with your installation's path and filename.

Install for Python 3.7 or 2.7 in the terminal:

For Python 3.7, enter the following:

.. code-block:: console

    # Include the bash command regardless of whether or not you are using the Bash shell
    bash ~/Downloads/Anaconda3-2020.05-Linux-x86_64.sh
    # Replace ~/Downloads with your actual path
    # Replace the .sh file name with the name of the file you downloaded

For Python 2.7, enter the following:

.. code-block:: console

    # Include the bash command regardless of whether or not you are using the Bash shell
    bash ~/Downloads/Anaconda2-2019.10-MacOSX-x86_64.sh
    # Replace ~/Downloads with your actual path
    # Replace the .sh file name with the name of the file you downloaded

Press Enter to review the license agreement. Then press and hold Enter to scroll.

Enter “yes” to agree to the license agreement.

Use Enter to accept the default install location, use CTRL+C to cancel the installation, or enter another file path to specify an alternate installation directory. If you accept the default install location, the installer displays PREFIX=/home/<USER>/anaconda<2/3> and continues the installation. It may take a few minutes to complete.

Note

Anaconda recommends you accept the default install location. Do not choose the path as /usr for the Anaconda/Miniconda installation.

Anaconda recommends you enter “yes” to initialize Anaconda Distribution by running conda init.

If you enter “no”, then conda will not modify your shell scripts at all. In order to initialize conda after the installation process is done, run the following commands:

.. code-block:: console

    # Replace <PATH_TO_CONDA> with the path to your conda install
    source <PATH_TO_CONDA>/bin/activate
    conda init

For more information, see the FAQ.

The installer finishes and displays, “Thank you for installing Anaconda<2/3>!”

Close and re-open your terminal window for the installation to take effect, or enter the command source ~/.bashrc to refresh the terminal.

You can also control whether or not your shell has the base environment activated each time it opens.

.. code-block:: console

    # The base environment is activated by default
    conda config --set auto_activate_base True

    # The base environment is not activated by default
    conda config --set auto_activate_base False

    # The above commands only work if conda init has been run first
    # conda init is available in conda versions 4.6.12 and later

Verify your installation.

Note

If you install multiple versions of Anaconda, the system defaults to the most current version, as long as you haven’t altered the default install path.

To use this package clone the repository and install the dependencies.

Create a new conda environment
------------------------------

.. code-block:: console

 $ conda create -n <env_name> python=3.7

Activate the environment

.. code-block:: console

 $ conda activate <env_name>

Create a new conda environment from an environment.yml file

.. code-block:: console

 $ git clone https://github.com/Pecneb/computer_vision_research.git

 $ conda env create -f environment.yml