# Copyright (c) 2019 Graphcore Ltd. All rights reserved.

#how to format recursively with clang-format, from
# https://stackoverflow.com/questions/28896909

copyright_return=$(python3 check_copyright.py)
if [[ "${copyright_return:0:1}" -gt 0 ]];
then 
echo "Missing copyright notices in one or several .hpp or .cpp files"
echo "${copyright_return}"
exit
fi

cf_version=$(python3 get_clang_format_version.py)
if [[ "${cf_version}" -lt 8 ]];
then 
echo "Clang-format version should be 8.0.0 or greater."
exit
fi

PROC_COUNT=9
printf "  -->  Inplace clang-formatting all .cpp and .hpp files\n"
find ../poprithms -iname *.[ch]pp | xargs -n 1 -P ${PROC_COUNT} clang-format -i -verbose


printf "\nFormatting complete.\n"

