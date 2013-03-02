import sys
import os
import io
from IPython.nbformat import current
 
 
def remove_outputs(nb):
    """Remove the outputs from a notebook"""
    i = 1
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.prompt_number = i
                i += 1
 
if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'clean':
        target = sys.argv[2]
        if os.path.isdir(target):
            fnames = [os.path.join(target, f)
                      for f in os.listdir(target)
                      if f.endswith('.ipynb')]
        else:
            fnames = [target]
        for fname in fnames:
            print("Removing outputs for: " + fname)
            with io.open(fname, 'rb') as f:
                nb = current.read(f, 'json')
            remove_outputs(nb)
            with io.open(fname, 'wb') as f:
                nb = current.write(nb, f, 'json')
    else:
        print("Unsupported command")
        sys.exit(1)