
For parsing output from Shimadzu GCMSSolutions software. To generate this format
of output from within the software, go to the `File->Export Data...` menu or
change your batch settings to include ASCII data export.

### Examples
```
import shimadzu
# Will give you a dict where each key is a sample name, and each value is
# another dict, containing the various information in the output.
samples = shimadzu.read('ASCIIData.txt')
```

### Notes
- Tables in the software are generally converted to pandas DataFrames.
- Most variables (column names, table names, etc.) are renamed to be closer to valid
Python identifiers.
- Tables that have a few tablewide properties are implemented as classes, where
  the table data is accessible at `<class_obj>.df`, and the properties are
  instance variables of the class.

