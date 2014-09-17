"""Utility script to be used to cleanup the notebooks before git commit

This a mix from @minrk's various gists.

"""

import time
import sys
import os
import io
try:
    from queue import Empty
except:
    # Python 2 backward compat
    from Queue import Empty

from IPython.nbformat import current
from IPython.kernel import KernelManager
from IPython.parallel import Client

try:
    FileNotFoundError
except NameError:
    FileNotFoundError = IOError


assert KernelManager  # to silence pyflakes


def remove_outputs(nb):
    """Remove the outputs from a notebook"""
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type == 'code':
                cell.outputs = []
                if 'prompt_number' in cell:
                    del cell['prompt_number']


def remove_signature(nb):
    """Remove the signature from a notebook"""
    if 'signature' in nb.metadata:
        del nb.metadata['signature']


def run_cell(shell, iopub, cell, timeout=300):
    if not hasattr(cell, 'input'):
        return [], False
    shell.execute(cell.input)
    # wait for finish, maximum 5min by default
    reply = shell.get_msg(timeout=timeout)['content']
    if reply['status'] == 'error':
        failed = True
        print("\nFAILURE:")
        print(cell.input)
        print('-----')
        print("raised:")
        print('\n'.join(reply['traceback']))
    else:
        failed = False

    # Collect the outputs of the cell execution
    outs = []
    while True:
        try:
            msg = iopub.get_msg(timeout=0.2)
        except Empty:
            break
        msg_type = msg['msg_type']
        if msg_type in ('status', 'pyin'):
            continue
        elif msg_type == 'clear_output':
            outs = []
            continue

        content = msg['content']
        out = current.NotebookNode(output_type=msg_type)

        if msg_type == 'stream':
            out.stream = content['name']
            out.text = content['data']
        elif msg_type in ('display_data', 'pyout'):
            for mime, data in content['data'].items():
                attr = mime.split('/')[-1].lower()
                # this gets most right, but fix svg+html, plain
                attr = attr.replace('+xml', '').replace('plain', 'text')
                setattr(out, attr, data)
            if msg_type == 'pyout':
                out.prompt_number = content['execution_count']
        elif msg_type == 'pyerr':
            out.ename = content['ename']
            out.evalue = content['evalue']
            out.traceback = content['traceback']
        else:
            print("unhandled iopub msg: %s" % msg_type)

        outs.append(out)

    # Special handling of ipcluster restarts
    if '!ipcluster stop' in cell.input:
        # wait some time for cluster commands to complete
        for i in range(10):
            try:
                if len(Client()) == 0:
                    break
            except FileNotFoundError:
                pass
            sys.stdout.write("@"); sys.stdout.flush()
            time.sleep(5)
    if '!ipcluster start' in cell.input:
        # wait some time for cluster commands to complete
        for i in range(10):
            try:
                if len(Client()) > 0:
                    break
            except FileNotFoundError:
                pass
            sys.stdout.write("#"); sys.stdout.flush()
            time.sleep(5)
    return outs, failed


def run_notebook(nb):
    km = KernelManager()
    km.start_kernel(stderr=open(os.devnull, 'w'))
    if hasattr(km, 'client'):
        kc = km.client()
        kc.start_channels()
        iopub = kc.iopub_channel
    else:
        # IPython 0.13 compat
        kc = km
        kc.start_channels()
        iopub = kc.sub_channel
    shell = kc.shell_channel

    # simple ping:
    shell.execute("pass")
    shell.get_msg()

    cells = 0
    failures = 0
    for ws in nb.worksheets:
        for cell in ws.cells:
            if cell.cell_type != 'code':
                continue

            outputs, failed = run_cell(shell, iopub, cell)
            cell.outputs = outputs
            cell['prompt_number'] = cells
            failures += failed
            cells += 1
            sys.stdout.write('.')
            sys.stdout.flush()

    print()
    print("ran notebook %s" % nb.metadata.name)
    print("    ran %3i cells" % cells)
    if failures:
        print("    %3i cells raised exceptions" % failures)
    kc.stop_channels()
    km.shutdown_kernel()
    del km


def process_notebook_file(fname, action='clean', output_fname=None):
    print("Performing '{}' on: {}".format(action, fname))
    orig_wd = os.getcwd()
    with io.open(fname, 'r') as f:
        nb = current.read(f, 'json')

    if action == 'check':
        os.chdir(os.path.dirname(fname))
        run_notebook(nb)
        remove_outputs(nb)
        remove_signature(nb)
    elif action == 'render':
        os.chdir(os.path.dirname(fname))
        run_notebook(nb)
    else:
        # Clean by default
        remove_outputs(nb)
        remove_signature(nb)

    os.chdir(orig_wd)
    if output_fname is None:
        output_fname = fname
    with io.open(output_fname, 'w') as f:
        nb = current.write(nb, f, 'json')


if __name__ == '__main__':
    # TODO: use argparse instead
    args = sys.argv[1:]
    targets = [t for t in args if not t.startswith('--')]
    action = 'check' if '--check' in args else 'clean'
    action = 'render' if '--render' in args else action

    rendered_folder = os.path.join(os.path.dirname(__file__),
                                   'rendered_notebooks')
    if not os.path.exists(rendered_folder):
        os.makedirs(rendered_folder)
    if not targets:
        targets = [os.path.join(os.path.dirname(__file__), 'notebooks')]

    for target in targets:
        if os.path.isdir(target):
            fnames = [os.path.abspath(os.path.join(target, f))
                      for f in os.listdir(target)
                      if f.endswith('.ipynb')]
        else:
            fnames = [target]
        for fname in fnames:
            if action == 'render':
                output_fname = os.path.join(rendered_folder,
                                            os.path.basename(fname))
            else:
                output_fname = fname
            process_notebook_file(fname, action=action,
                                  output_fname=output_fname)
