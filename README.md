# iQuCodeFest 2026 - Quantum ÉTS Team A
pour executer le code, faire uv sync pour syncroniser les paquets requis, 
à partir du root du repo, faire `python src/quanvolution_layer.py` --preprocess pour généré les données
faire `python src/quanvolution_layer.py` pour juste afficher les données 

pour l'execution sur le hardware, créer un fichier .env avec CRN et TOKEN, voir .env.example

`src\hardware_execution.py --preprocess --run-id quebec_parallel_001`

### Section inspiré de 2604.07639

le fichier /notebook/QOS-sketching.py contients l'expérience
controles : +, -, 1, 2, 3, 4

L’idée du programme est la suivante : on prend l’image complète, on choisit aléatoirement k positions de pixels, puis on ne conserve que ces k pixels pour les encoder en FRQI. Ensuite, lors de la reconstruction, on remplit les pixels restants à l’aide d’un algorithme du plus proche voisin.