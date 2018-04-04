# Changes all the __DYNAMIC__ variables in Python files to be True/False

IS_FALSE=$(find . -iname '*.py' | xargs grep -E "^__DYNAMIC__ = False")
if [ "$IS_FALSE" == "" ]; then
    find . -iname '*.py' | xargs sed -i "s/__DYNAMIC__ = True/__DYNAMIC__ = False/g"
    echo "dynamic is now False"
else
    find . -iname '*.py' | xargs sed -i "s/__DYNAMIC__ = False/__DYNAMIC__ = True/g"
    echo "dynamic is now True"
fi
