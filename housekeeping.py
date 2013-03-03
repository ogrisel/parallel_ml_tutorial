"""Utility script to be used to cleanup the notebooks before git commit"""

import sys
import os
import io
from IPython.nbformat import current


def remove_outputs(nb):
    """Remove the outputs from a notebook"""
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                cell.prompt_number = None


def remote_solutions(nb):
    """Generate a version of the notebook with stripped exercises solutions"""
    for ws in nb.worksheets:
        inside_solution = False
        cells_to_remove = []
        for i, cell in enumerate(ws.cells):
            if cell.cell_type == 'heading':
                inside_solution = False
            elif cell.cell_type == 'markdown':
                first_line = cell.source.split("\n")[0].strip()
                if first_line.lower() in ("**exercise:**", "**exercise**:"):
                    inside_solution = True
                    # Insert a new code cell to work on the exercise
                    ws.cells.insert(i + 1, current.new_code_cell())
                    continue
            if inside_solution:
                if cell.cell_type == 'code' and not hasattr(cell, 'input'):
                    # Leave blank code cells
                    continue
                cells_to_remove.append(cell)
        for cell in cells_to_remove:
            ws.cells.remove(cell)


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
    elif cmd == 'exercises':
        # Generate the notebooks without the exercises solutions
        fnames = [f for f in os.listdir('solutions')
                  if f.endswith('.ipynb')]
        for fname in fnames:
            solution = os.path.join('solutions', fname)
            notebook = os.path.join('notebooks', fname)
            print("Generating solution-free notebook: " + notebook)
            with io.open(solution, 'rb') as f:
                nb = current.read(f, 'json')
            remote_solutions(nb)
            remove_outputs(nb)
            with io.open(notebook, 'wb') as f:
                nb = current.write(nb, f, 'json')
    else:
        print("Unsupported command")
        sys.exit(1)