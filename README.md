# <img src="firecrest_emblem3002.png" width="60"> firecrest

### Installing the __firecrest__ package

```
git clone git@github.com:Corwinpro/firecrest.git
cd firecrest
pip install --user -e .
```

### Setting up the __dolfin__ environment

Install the dolfin and necessary packages and activate:

```
conda create -n fenicsproject -c conda-forge fenics matplotlib meshio
source activate fenicsproject
```

Deactivate:
```
source deactivate fenicsproject
```

### Additional packages

Install __pysplines__ [from here](https://github.com/Corwinpro/PySplines).

<sup>Logo by [Titov Fedor](https://www.artstation.com/quietvictories)</sup>
