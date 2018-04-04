find . -iname '*.py' | xargs eval $(grep -Eh "a.autogen_init")

#for cmd in $FOUND; do
#    echo "cmd = $cmd"
#done
# find . -iname '*.pyx'

