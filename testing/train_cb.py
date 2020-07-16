import pdb
import sys
import subprocess
import argparse

p = argparse.ArgumentParser()
p.add_argument('model_file', help="model specification", default="cb_hand2_v_shuffle_3areas")
p.add_argument('-g', '--gpus', nargs='?', type=int, default=1)
p.add_argument('-s', '--seed', nargs='?', type=int, default=100)
p.add_argument('-lambdar', '--lambdar', type=float, default=0)
p.add_argument('-lambdaw', '--lambdaw', type=float, default=1)
p.add_argument('-clean', '--clean', type=bool, default=False)
p.add_argument('-suffix', '--suffix', nargs='?', type=str, default=' ')
p.add_argument('-lr', '--lr', type=float, default=5e-5)
# a is a class and a.model_file is the model
a = p.parse_args()


def call(s):
    rv = subprocess.call(s.split())
    if rv != 0:
        sys.stdout.flush()
        print("Something went wrong (return code {}).".format(rv)
              + " We're probably out of memory.")
        sys.exit(1)


if a.clean:
    call("python ../examples/do.py ../examples/models/{} clean --suffix {}".format(a.model_file, a.suffix))


# train
call("python ../examples/do.py ../examples/models/{}.py train -s {} -g {} -lr {} -lambdar {} -lambdaw {} --suffix {}".format(a.model_file, a.seed, a.gpus, a.lr, a.lambdar, a.lambdaw, a.suffix))
# call("python ../examples/do.py ../examples/models/{}.py train -s {} -g {}".format(a.model_file, a.seed, a.gpus))

# structure
#call("python ../examples/do.py ../examples/models/{}.py structure -s 1".format(a.model_file))
