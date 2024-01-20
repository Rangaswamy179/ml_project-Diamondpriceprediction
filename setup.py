from setuptools import find_packages,setup
from typing import List

HYPEN_E_DOT="-e ."

def get_requirement(file_path:str)->List[str]:
  requirements=[]
  with open(file_path) as file_obj:
    requirements=file_obj.readlines()
    requirements=[r.replace("\n","") for r in requirements]
    if HYPEN_E_DOT in requirements:
      requirements.remove(HYPEN_E_DOT)

    return requirements
  
setup(
  name='Diamond',
  version='0.0.1',
  author='r',
  author_email='e@gmail.com',
  install_requires=get_requirement('requirements.txt'),
  packages=find_packages()
)