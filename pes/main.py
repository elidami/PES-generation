from highsym import get_slab_hs, get_interface_hs, pbc_hs_points, fix_hs_dicts
from monty.json import jsanitize
from manipulate_struct import flip_slab
from pymatgen.io.vasp import Poscar
from generate_pes_input import generate_pes_inputs
import os, sys, shutil, json
from potensurf import get_pes
import numpy as np
from mep import MEP
from pymatgen.io.vasp.outputs import Outcar


def generate_hs(poscar_top_path, poscar_bot_path):
    """
    Generate high symmetry points (HS) for an interface between two slabs.

    Parameters:
    poscar_top_path (str): The file path of the top slab POSCAR file.
    poscar_bot_path (str): The file path of the bottom slab POSCAR file.

    Returns:
    dict: A dictionary containing the high symmetry points for the interface.

        The dictionary has the following keys:
        - 'bot_unique': irreducible high symmetry points of the bottom slab.
        - 'bot_all': reducible high symmetry points of the bottom slab.
        - 'top_unique': irreducible high symmetry points of the top slab.
        - 'top_all': reducible high symmetry points of the top slab.
        - 'inter_unique': irreducible high symmetry points of the interface.
        - 'inter_all': reducible high symmetry points of the interface.
    """

    #Retrieve the structure of top and bottom slabs
    poscartop = Poscar.from_file(poscar_top_path)
    top_structure = poscartop.structure

    poscarbot = Poscar.from_file(poscar_bot_path)
    bot_structure = poscarbot.structure

    flipped_top = flip_slab(top_structure)

    # Get the high symmetry points
    top_hsp_unique, top_hsp_all = get_slab_hs(flipped_top)
    bottom_hsp_unique, bottom_hsp_all = get_slab_hs(bot_structure)

    cell = bot_structure.lattice.matrix

    hsp_unique = get_interface_hs(bottom_hsp_unique, top_hsp_unique, cell) 
    hsp_all = get_interface_hs(bottom_hsp_all, top_hsp_all, cell)

    c_hsp_u, c_hsp_a = fix_hs_dicts(hsp_unique, hsp_all, top_structure, bot_structure)

    b_hsp_u =  pbc_hs_points(bottom_hsp_unique, cell)
    b_hsp_a =  pbc_hs_points(bottom_hsp_all, cell)
    t_hsp_u =  pbc_hs_points(top_hsp_unique, cell)
    t_hsp_a =  pbc_hs_points(top_hsp_all, cell)

    # Create a HS dictionary to be stored in the Database
    hs = {
        'bot_unique': b_hsp_u,
        'bot_all': b_hsp_a,
        'top_unique': t_hsp_u,
        'top_all': t_hsp_a,
        'inter_unique': jsanitize(c_hsp_u),
        'inter_all': jsanitize(c_hsp_a)}
    
    # Write the HS dictionary to a JSON file called 'hs.json'
    with open('./hs.json', 'w') as f:
        json.dump(hs, f, indent=4)

    return hs

def write_input(output_folder, structures, shifts):
    """
    Write input files for each structure and shift.

    Parameters:
    output_folder (str): The path to the output folder where the input files will be written.
    structures (list): A list of structures.
    shifts (list): A list of shifts.

    Returns:
    None
    """

    for i,(shift, struct) in enumerate(zip(shifts, structures)):
        # Crea una sottocartella per ogni shift
        folder = os.path.join(output_folder, str(shift))
        os.makedirs(folder, exist_ok=True)

        # Scrivi il file POSCAR nella sottocartella
        selective_dynamics = [[False, False, True] for _ in struct]
        poscar = Poscar(struct,selective_dynamics=selective_dynamics)
        poscar.write_file(os.path.join(folder, 'POSCAR'))

        # Copy the INCAR, KPOINTS, POTCAR and jobscript files to the subfolder
        #for file in ['INCAR', 'KPOINTS', 'POTCAR', 'jobscript']:
         #   shutil.copyfile(file, os.path.join(folder, file))

def launch_jobs_in_subfolders(main_folder, jobscript):
    """
    Launches jobs in subfolders of the main_folder using the specified jobscript.

    Parameters:
    main_folder (str): The path to the main folder containing subfolders.
    jobscript (str): The name of the jobscript to be executed.

    Returns:
    None
    """

    # List of all subfolders in main_folder
    subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]

    # Save the current directory
    current_folder = os.getcwd()

    # Execute jobs
    for folder in subfolders:
        os.chdir(folder)  
        os.system(f'sbatch {jobscript}') 
        os.chdir(current_folder) 

    # Return to the original directory
    os.chdir(current_folder)


def extract_total_energy(outcar_path):
    outcar = Outcar(outcar_path)
    return outcar.final_energy


def extract_energies_and_write_to_json(main_folder, hs_unique, output_file):
    """
    Extracts energies from OUTCAR files in the specified main_folder directory and writes the results to a JSON file.

    Parameters:
    - main_folder (str): The path to the main folder containing subfolders with OUTCAR files.
    - hs_unique (dict): A dictionary containing the unique shift labels and their corresponding xy coordinates.
    - output_file (str): The path to the output JSON file.

    Returns:
    None
    """
    # Ottieni un elenco di tutte le sottocartelle in main_folder
    subfolders = [f.path for f in os.scandir(main_folder) if f.is_dir()]

    results = {'hs_unique': []}
    for folder in subfolders:
        # Costruisci il percorso del file OUTCAR
        outcar_path = os.path.join(folder, 'OUTCAR')
        
        # Estrai la total energy dal file OUTCAR
        energy = extract_total_energy(outcar_path)  

        # Estrai la label dello shift dal nome della cartella
        label = os.path.basename(folder)

        # Trova lo shift corrispondente in hs_unique
        for key, value in hs_unique.items():
            if key == label:
                # Aggiungi un nuovo array con la label, le coordinate xy e l'energia
                results['hs_unique'].append([label, value[0][0], value[0][1], energy])

    # Scrivi i risultati in un file JSON
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)

def main():

    # Read settings from json file
    with open('./settings.json', 'r') as f:
        settings = json.load(f)
    
    output_folder = './output'
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    command = sys.argv[1]

    if command == 'gen':

        # Calculate high symmetry points
        print("Calculating high symmetry points...")
        hs = generate_hs(settings['path_to_top_slab'], settings['path_to_bottom_slab'])
        hs_unique = hs['inter_unique']
        hs_all = hs['inter_all']

        print("Generating PES input...")
        
        # Retrieve top and bottom slabs structures. In the settings you choose if you want to use the decorated one.
        if settings['decorated']:
            poscarbot = Poscar.from_file(settings['path_to_decorated_bottom'])
        else:
            poscarbot = Poscar.from_file(settings['path_to_bottom_slab'])
        
        bot_slab = poscarbot.structure  

        poscartop = Poscar.from_file(settings['path_to_top_slab'])
        top_slab = poscartop.structure

        # Generate PES inputs
        structure = generate_pes_inputs(bot_slab, top_slab, hs_unique)

        # Write input files
        write_input(output_folder, structure, hs_unique)
        print("Input files Generated.")
        
        # Launch jobs
        #launch_jobs_in_subfolders('./output', 'jobscript')
        


    elif command == 'plot':
        
        # Read the high symmetry points from the json file created in the previous operation
        with open('./hs.json', 'r') as f:
            data = json.load(f)
            hs_unique = data['inter_unique']
            hs_all = data['inter_all']

        # Extract the total energy from the OUTCAR file corresponding to the
        #inequivalent shifts and write the results to a JSON file
        extract_energies_and_write_to_json(output_folder, hs_unique, './results.json')

        print("Retrieved the total energy of the inequivalent shifts.")

        pes_name = settings['pes_name']
        plot_name = 'PES_' + pes_name.replace('(', '').replace(')', '') + '.png'
        colorbar_limit=settings['pes_max_value']

        poscartop = Poscar.from_file(settings['path_to_top_slab'])
        top_slab = poscartop.structure  
        cell = top_slab.lattice.matrix

        # Extract the lattice vectors from the lattice matrix
        a = np.array(cell[0])
        b = np.array(cell[1])

        # Area of the interface supercell
        area = np.linalg.norm(np.cross(a, b))
        print(f"Area of the interface supercell: {area} A^2")

        # Convert the enrgies of the unique shifts from eV to J/m^2
        with open('./results.json', 'r') as f:
            data_results = json.load(f)

        hs_unique_energy = data_results['hs_unique']
        for item in hs_unique_energy:
            item[3] = item[3] * 16.0218 / area


        #nel codice originale get_pes ha come input hs_all sputati fuori dall'FT precedente (solo coordinate) e hs_unique_Jm2 (quindi contenenti anche l'info dell'energia in J/m^2)
            
        # Plot the PES
        print("Performing the interpolation...")

        import time
        start_time = time.time() 
        solver, v_list_jm2, data_jm2, rbf = get_pes(hs_all, hs_unique_energy, cell, 
                                  title=pes_name, to_fig=plot_name, colorbar_limit=colorbar_limit)
        #data_jm2: coordinate+energie (col min a 0!!)
        print("PES plot generated.")
        end_time = time.time() 
        elapsed_time_pes = end_time - start_time  # Calcola il tempo trascorso
        print(f"PES time: {elapsed_time_pes} seconds") 
        
        data = {
                'all_values': jsanitize(v_list_jm2),
                'pes_grid': jsanitize(data_jm2),
                }
        
        with open('./data_for_mep.json', 'w') as f:
            json.dump(data, f, indent=4)
            
        
        # Calculate the shear strength
        #mep = MEP(rbf=params['rbf'], cell=params['pes_grid'])
        mep = MEP(rbf=rbf, cell=cell) 
        print("Object MEP created.")
        print("Computing the MEP...")

        start_time = time.time() 
        mep_line, mep_data = mep.get_mep(method='zerotemp', optimization='bs_line')
        print(mep_line)

        solver.plot(extent=(2, 2), mpts=(200j, 200j), title="mep", to_fig="mep.png", colorbar_limit=colorbar_limit, mep=mep_line)
        #print(f"mep_line: {mep_line}")
        #print(f"mep_data: {mep_data}")
        print("MEP completed.")
        end_time = time.time() 
        elapsed_time_mep = end_time - start_time  
        print(f"MEP time: {elapsed_time_mep} seconds")
        
        shear_strength, data_ss_xy = mep.get_shear_strenght(delta=0.01)
        print(f"Shear strength: {shear_strength} GPa")
        #print(f"data_ss_xy: {data_ss_xy}")



    else:
        print("Command not recognized.")















