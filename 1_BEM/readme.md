# Group 37 - Assignment #1 - BEM

Assignment choice: **Propeller**

Project members:
- Kasper
- Greg Kaminski
- Alfonso Medina Marrero

The project structure is as follows:
==== Folders
├───/data             : stores the airfoil data
├───/plots            : stores the plots
├───/results          : stores numerical results
├───/verification     : stores JavaProp verification data (ARAD6 propeller)
==== Main Physics Code
├───Annuli.py         : contains BEM code for a single annuli
├───Rotor.py          : discretizes blade and evaluates integrated properties
├───tip_correction.py : contains tip correction codes
├───multi_rotor_analysis.py : evaluates rotors for different conditions (e.g. varying J)
==== Post-Processing Code
├───verification.ipynb : compares our BEM code with Ning and JavaProp
├───plotting_routines.py : contains plotting functions
├───plotting_sandbox.ipynb : creates results plot for the assingment questions
==== Post-Processing Code
└───optimizer.py: perform optimization to maximize power generation

Python Requirements:
- matplotlib
- pandas
- numpy
- scipy
- notebook
- jupyter
- juliacall (to verify against Ning Code)

Julia requirements
- CCBlade (to verify against Ning Code)
