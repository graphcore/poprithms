Running the regression suite
----------------------------

# where the log file will be written
mkdir my_log_dir

#where the plots (.pdfs) will be saved
mkdir my_plot_dir

# generate the log file 
./path_to_regression_executable my_log_dir/my_log_file.log

# generate the plots files
python3 path_of_summarize.py my_log_dir

This will create several pdfs - one for each Graph to be scheduled. 

Note that it is possible to write several log files to my_log_dir. 
For example, log files from  different branches/commits can be written to 
my_log_dir. This is useful to detect for performance regressions/improvements. 

