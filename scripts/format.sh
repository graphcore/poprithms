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


header_guard_spacing_return=$(python3 check_header_guard_spacing.py)
if [[ "${header_guard_spacing_return:0:1}" -gt 0 ]];
then 
echo "Incorrect header guard spacing in one or several .hpp files"
echo "${header_guard_spacing_return}"
exit
fi

# number of threads to run clang-format with
PROC_COUNT=9

# if there is no program gc-clang-format, check for a clang-format version 13 or
# greater. Otherwise, use gc-clang-format. gc-clang-format is generally available as
# as soon as you have sourced the poplar view build activation script. gc-clang-format
# is the pinned version of clang-format which poplar and poplibs use.
if ! command -v gc-clang-format &> /dev/null
then
cf_version=$(python3 get_clang_format_version.py)
if [[ "${cf_version}" -lt 14 ]];
then 
echo "gc-clang-format, or a clang-format version greater than 14.0.0, must be used to format poprithms C++ code."
exit
fi
printf "  -->  Inplace clang-formatting all .cpp and .hpp files\n"
find ../poprithms -iname *.[ch]pp | xargs -n 1 -P ${PROC_COUNT} clang-format -i -verbose

# if there is a gc-clang-format available, use it. 
else
printf "  -->  Inplace clang-formatting all .cpp and .hpp files\n"
find ../poprithms -iname *.[ch]pp | xargs -n 1 -P ${PROC_COUNT} gc-clang-format -i -verbose
fi

printf "\nFormatting complete.\n"


