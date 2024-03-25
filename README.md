# PES-generation
This code is divided into two tasks:
1. **Generation and launch of VASP jobs for the Potential Energy Surface (PES) computation**. This works for clean and decorated (i.e. with intercalated atoms between the surfaces) interfaces. This module works by providing the structures of upper and bottom slabs, for which the high symmetry points are calculated. Then all the relative shifts are computed and symmetrically equivalent structures are deleted. Finally, VASP calculations are launched for all the inequivalent interfaces; in particular, relaxation calculation along the z direction are performed. Note that, even if the high symmetry points are calculated for the surfaces without any adosrbed atom, VASP calculations can be performed for the decorated interfaces: you only need to specify that in the settings and provide the decorated bottom slab structure (see next sections for details). After the execution of this part, input and output files of every VASP calculation are organized in the `output` folder and the `hs.json` file is written: it contains all the information about the reducible and irreducible high symmetry points of both slabs and of the interface.
2. **Creation of PES images for decorated and clean interfaces**. This second module computes the PES image based on the energy values associated to each inequivalent shift, obtained in the previous step. Additionally, the `results.json` file is created, containing the coordinates of the unique shifts and their associated energies. 

## Installation 
First, clone the repository into your local machine:

`git clone https://github.com/elidami/PES-generation.git`

Once the download is completed, go inside the downloaded folder and run:

`bash install.sh`

to add the executable in the PATH variable. With this operation, you can launch the program from any folder.

## How to run the code
First, go inside your working folder. This must contain some VASP input files, such as INCAR, POTCAR, KPOINTS and the jobscript file. 

In order to generate the input structures for the PES computation and to launch the VASP jobs, run the command:

`pes gen`

Then, for the generation of the PES image, run the command:

`pes plot`

The input parameters, such as the path to the bottom and top slabs structures, must be specified in the `settings.json` file. See next section for details.

## Input parameters
As mentioned, all input parameters are specified in the `settings.json` file:  
* `decorated`: can be True or False. Set it to True if you want to compute the PES of a decorated interface.
* `pes_name`: "str". Choose the name of the interface (it will be written in the title of the PES image).
* `path_to_bottom_slab`: "path to the structure of the bottom slab". It must be written in POSCAR format.
* `path_to_top_slab`: "path to the structure of the top slab". It must be written in POSCAR format.
* `path_to_decorated_bottom`: "path to the structure of the decorated bottom slab". It is not mandatory if you want to calculate the PES of a clean interface.
* `pes_max_value`: maximum value for the energy in the colorbar of the PES image. Set it to `null` if you want that the maximum value in the colorbar corresponds to the max value of the PES energy. 


