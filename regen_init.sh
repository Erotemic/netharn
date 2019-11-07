SOURCE=$(find . -iname '*.py' | xargs grep -Eh "a.autogen_init")
echo "SOURCE = $SOURCE"
sh -c "$SOURCE"

#for cmd in $FOUND; do
#    echo "cmd = $cmd"
#done
# find . -iname '*.pyx'

