#!/usr/bin/env python
import bpf4
from emlib import doctools
import os
from pathlib import Path

 
def findRoot():
    p = Path(__file__).parent
    if (p/"index.md").exists():
        return p.parent
    if (p/"setup.py").exists():
        return p
    raise RuntimeError("Could not locate the root folder")
        

def main(dest: Path):
    config = doctools.RenderConfig(splitName=True, includeInheritedMethods=False)
    modules = {'core': bpf4.core, 
               'util': bpf4.util, 
               'api': bpf4.api}

    for name, module in modules.items():
        docs = doctools.generateDocsForModule(module, renderConfig=config, title=name,
                                              includeCustomExceptions=False)
        open(dest/f"{name}.md", "w").write(docs)
    
    
if __name__ == "__main__":
    root = findRoot()
    docsfolder = root / "docs"
    print("Documentation folder", docsfolder)
    assert docsfolder.exists()
    main(docsfolder)
    os.chdir(root)
    os.system("mkdocs build")
