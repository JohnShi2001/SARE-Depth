"""Compare a fresh example evaluation with the measured reference output."""
import argparse
import json
import math
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('summary',type=Path)
    p.add_argument('--rtol',type=float,default=1e-4)
    p.add_argument('--atol',type=float,default=1e-6)
    a=p.parse_args()
    actual=json.loads(a.summary.read_text())
    expected=json.loads((Path(__file__).parent/'results/example_cpu_expected.json').read_text())
    assert actual['seed']==expected['seed'], 'Example checkpoint differs'
    assert actual['protocol']==expected['protocol'], 'Evaluation protocol differs'
    for stage,values in expected['metrics'].items():
        for name,value in values.items():
            if isinstance(value,int):
                assert actual['metrics'][stage][name]==value,(stage,name)
            else:
                assert math.isclose(actual['metrics'][stage][name],value,rel_tol=a.rtol,abs_tol=a.atol),(stage,name,actual['metrics'][stage][name],value)
    print('PASS: example count, support and metrics agree within stated tolerances')


if __name__=='__main__':main()
