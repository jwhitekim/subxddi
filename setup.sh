pip install torch==2.3.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install dgl -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
pip install "numpy<2"
pip install torchdata==0.7.1
pip install pyyaml pydantic
pip install networkx

python -c "import torch; import dgl; print('torch:', torch.__version__); print('dgl:', dgl.__version__)"