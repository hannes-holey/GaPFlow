#! /bin/sh
# Updates all Python files with license and copyright information obtained from the git log.

for fn in `find GaPFlow/*.py \
                GaPFlow/md \
                GaPFlow/models \
                GaPFlow/cli \
                GaPFlow/viz \
                tests -name "*.py"`; do
  echo $fn
  python3 maintenance/copyright.py $fn | cat - LICENSES/MIT.txt | python3 maintenance/replace_header.py $fn
done
