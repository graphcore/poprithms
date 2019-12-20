#how to format recursively with clang-format, from
# https://stackoverflow.com/questions/28896909

cf_version=$(python3 get_clang_format_version.py)
if [[ "${cf_version}" -lt 8 ]];
then 
echo "Clang-format version should be 8.0.0 or greater."
exit
fi

printf "  -->  Inplace clang-formatting all .cpp files\n"
find ../poprithms -iname *.cpp | xargs clang-format -i -verbose

printf "  -->  Inplace clang-formatting all .hpp files\n"
find ../poprithms -iname *.hpp | xargs clang-format -i -verbose

printf "\nFormatting complete.\n"
