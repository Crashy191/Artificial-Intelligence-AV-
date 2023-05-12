import torch

if torch.cuda.is_available():
    print('CUDA está instalado correctamente.')
else:
    print('No se pudo detectar CUDA en el sistema.')