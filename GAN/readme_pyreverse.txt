Install pylint with "pip install pylint"
Create environment variable PYTHONPATH with the path to the directory containing the .py files

Install Graphviz using this guide: https://bobswift.atlassian.net/wiki/spaces/GVIZ/pages/20971549/How+to+install+Graphviz+software

Use these commands:

pyreverse -S main.py gan.py generator_util.py discriminator_util.py layers.py util.py
dot -Tpdf classes.dot -o output.pdf