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
    core = doctools.generateDocsForModule(bpf4.core, renderConfig=renderConfig, 
                                          title="Core",
                                          includeCustomExceptions=False)
    open(dest/"core.md", "w").write(core)

    utildocs = doctools.generateDocsForModule(bpf4.util, renderConfig=renderConfig,
                                              title='Util')
    open(dest/'util.md', 'w').write(utildocs)
    apidocs = doctools.generateDocsForModule(bpf4.api, renderConfig=renderConfig,
                                             title='API')
    open(dest/'api.md', 'w').write(apidocs)
    
if __name__ == "__main__":
    root = findRoot()
    docsfolder = root / "docs"
    assert docsfolder.exists()
    main(docsfolder)
    os.chdir(root)
    os.system("mkdocs build")
