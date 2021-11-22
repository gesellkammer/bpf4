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
        

def main(destfolder: str):
    renderConfig = doctools.RenderConfig(splitName=True, fmt="markdown", docfmt="markdown")
    dest = Path(destfolder)
    reference = doctools.generateDocsForModule(bpf4.core, 
                                               renderConfig=renderConfig, 
                                               # exclude={'MidiBase'},
                                               title="Core")
    open(dest / "core.md", "w").write(reference)
    
    
if __name__ == "__main__":
    root = findRoot()
    docsfolder = root / "docs"
    assert docsfolder.exists()
    main(docsfolder)
    os.chdir(root)
    os.system("mkdocs build")
