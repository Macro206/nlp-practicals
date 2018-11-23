# NLP Practicals

This git repository contains the scripts I've written as part of my Part II Units of Assessment NLP practicals.

## Part I

Please use python2 when running this code. You will also require the SVMlight package - check the Part I report for a link to it (or use `pip install svmlight`).

If testing using stemmed reviews, please create the necessary directories and run `perform_stemming.py` to generate the stemmed reviews before trying to run the other scripts with the appropriate options.

## Part II

Please use python2 when running this code. You will also require the gensim package, as well as numpy and svmlight.

See the footnote in the report for a link to download the 100,000 review corpus. The `generate_file_list.py` script in the `tokenization` directory defines functions to move the reviews from the subdirectories of the decompressed folder into a single folder, and generate a file list to be used by the tokenisation command (see `tokenization/tokenization_command.txt`). Be sure to replace all paths in the python script and the command to the appropriate ones for your system. Furthermore, be sure to execute the tokenisation command from the directory containing Stanford CoreNLP on your system.

When running `doc2vec_classifier.py` and using it to train models, be sure to use `importDataTokenized()` as opposed to `importData()` (this is the default in `generateDoc2VecModel()`). Also be sure to change the folder path in there to your tokenised data folder.

Be sure to also replace the relevant paths in `review_loader.py`. Furthermore, set up a directory for your Amazon data - make a folder for each category that has the same name as its corresponding JSON file (without the file extension), and add the JSON file to that folder. Then use the `amazon_data_processing.py` script to automatically choose samples and tokenise all of the data (but be sure to set the appropriate paths in there).

Finally, the `cross_validation.py` file contains most of the code that will actually produce results. However, be aware that you should invoke whatever function you need at the bottom of the file - there is no "main" method that will re-create all of my results. A lot of the function calls I used are commented out at the bottom (including some that produce results that I didn't include in my final report) - use these as a guide.

