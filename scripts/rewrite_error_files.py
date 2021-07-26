# Copyright (c) 2021 Graphcore Ltd. All rights reserved.
from pathlib import Path
import os


def rewriteTheErrorFiles():
    """
    Rewrite all error.hpp and error.cpp files using a specific template.
    This script should not be used regularly. It was written to help with a 
    significant change in the design of the error files, and is being stored in 
    the repo in case it is useful for other similar tasks down the road. 
    """

    #find all error.hpp file names (excluding error/error.hpp)
    headers = list(Path("../poprithms").rglob("error.hpp"))
    headers = [x for x in headers if "error/error" not in str(x)]

    for h in headers:

        # rewrite the header files #
        # ######################## #
        fn = str(h)
        x = fn.split("poprithms/poprithms/src/poprithms")[-1].split(
            "error.hpp")[0]
        parts = x[1:-1].split('/')
        guard = "POPRITHMS_" + "_".join([x.upper()
                                         for x in parts]) + "_ERROR_HPP"

        x = "// Copyright (c) 2021 Graphcore Ltd. All rights reserved.\n"
        x += "#ifndef %s\n" % (guard, )
        x += "#define %s\n\n" % (guard, )
        x += "#include <poprithms/error/error.hpp>\n"
        x += "\n\nnamespace poprithms {\n"
        for p in parts:
            x += ("namespace " + p + " { \n")

        x += "\n\npoprithms::error::error error(const std::string &what);\n"
        x += "poprithms::error::error error(uint64_t id, const std::string &what);\n\n\n}"

        for p in parts:
            x += "}"

        x += "\n\n#endif"
        filly = open(fn, "w")
        filly.write(x)
        filly.close()

        # rewrite the source file #
        # ########################
        #
        nsopen = "namespace poprithms {\n"
        nsclose = "}\n"
        for p in parts:
            nsopen += ("namespace " + p + "{\n")
            nsclose += "}\n"

        ns = '\"' + "::".join(parts) + '\"'
        incl = "/".join(parts)
        x = r"""// Copyright (c) 2021 Graphcore Ltd. All rights reserved.
    #include <%s/error.hpp>
    
    %s
    
    namespace {
    constexpr const char *const nspace(%s);
    }
    
    poprithms::error::error error(const std::string &what) {
      return poprithms::error::error(nspace, what);
    }
    
    poprithms::error::error error(uint64_t id, const std::string &what) {
      return poprithms::error::error(nspace, id, what);
    }
    
    
    %s
    """ % (incl, nsopen, ns, nsclose)

        print(x)

        filly = open(fn.replace("hpp", "cpp"), "w")
        filly.write(x)
        filly.close()
