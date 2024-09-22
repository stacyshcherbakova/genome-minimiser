# Generative for Genomic Sequence Generation 
## Abstract 
Designing minimal bacterial genomes remains a key challenge in synthetic biology, which has many industrial, pharmaceutical, and research applications. There is currently a lack of efficient tools for rapidly generating these minimal genomes, limiting their broader industrial adoption. Additionally, plasmids are essential tools in synthetic biology and biotechnology, enabling bioproduct research and production, yet plasmid backbone design is often overlooked, potentially limiting system optimisation. This project explores generative models for bacterial genomic sequence generation, focusing on two specific applications: 1) the generation of minimal genomes using variational autoencoders and 2) the generation of new plasmid designs using a generative adversarial network. The project results have shown that variational autoencoders can successfully create minimised genomes with most of the essential genes identified in the literature. Further, generative adversarial networks can be effectively applied to plasmid design. This study proposes a rapid, machine-learning-based approach for bacterial sequence generation, intending to accelerate genomic design processes and address the need for more efficient tools in synthetic biology.

## Some Acronyms/File names:
- **Extras**: Extra functions
- **Training**: Files which include training loops
- **Directories**: Directories paths
- **BD**: Horesh _et. al._ dataset (±7.5k samples)
- **Final**: Cut down Horesh _et. al._ dataset (±5k samples)
- **GS**: Gradient Clipping and Scheduling
- **AHPT**: After Hyperparameter Tuning

## Requirements
- **Python**: All required Python packages can be found in the `requirements.txt` file (for `pip` installation).
- **Conda**: The environment configuration, including Conda packages, is listed in the `environment.yml` file.
- **R**: For R scripts, the necessary packages are listed in `installed_packages.txt`.

To set up **pLannotate**, follow the instructions provided in their [repository](https://github.com/mmcguffi/pLannotate).


