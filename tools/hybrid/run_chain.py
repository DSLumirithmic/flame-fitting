''' Run a list of commands (JSON file: [[name, [argv...]], ...]) one after the other, logging to <log>; stops on failure. '''
import json, os, subprocess, sys, time
cmds = json.load(open(sys.argv[1]))
for name, argv in cmds:
    t0 = time.time()
    print('>>', name, ' '.join(argv), flush=True)
    r = subprocess.run(argv, cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), capture_output=True, text=True)
    print(r.stdout[-6000:], r.stderr[-4000:], flush=True)
    print('<<', name, 'rc', r.returncode, '%.1f s' % (time.time() - t0), flush=True)
    if r.returncode:
        print('CHAIN STOPPED', flush=True)
        sys.exit(1)
print('CHAIN DONE', flush=True)
