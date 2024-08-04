---
title: "Basics of time series, and adaptive conformal inference"
Date: 01 Aug 2024
output:
  html_document:
    df_print: paged
editor_options: 
  markdown: 
    wrap: 72
---



## Getting data

For the purposes of this blog, we will be using the richly structured `Turbofan Engine Degradation Simulation-2` data published along with other interesting industrial datasets on NASA's [prognostics for intelligent systems web page](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/). The data is made available as a zip file which we will need to fetch and extract. 


``` bash
# pwd
cd ../../../../../datasets
# mkdir turbofansimdata
cd turbofansimdata
# wget https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip
# for file in *; do
#    if [ -f "$file" ]; then
#        # Convert to lowercase, remove special chars, capitalize first letter of each word
#        newname=$(echo "$file" | tr '[:upper:]' '[:lower:]' | sed -e 's/[^a-zA-Z0-9]/ /g' -e 's/\b\(.\)/\u\1/g' | tr -d ' ')
#
#        # Rename the file if the new name is different
#        if [ "$file" != "$newname" ]; then
#            mv "$file" "$newname"
#            echo "Renamed: $file -> $newname"
#        fi
#    fi
# done
# mv 17TurbofanEngineDegradationSimulationDataSet2Zip TurbofanEngineDegradationSimulationDataSet2.Zip
# unzip *
# cd 17.\ Turbofan\ Engine\ Degradation\ Simulation\ Data\ Set\ 2/
# unzip *
# mkdir ../dataset
# mv data_set/* ../dataset/
# cd ..
# rm -rf 17.\ Turbofan\ Engine\ Degradation\ Simulation\ Data\ Set\ 2/
ls -l --block-size=M dataset
```

```
## total 27822M
## -rw-r--r-- 1 prasanna prasanna 2741M Jan 13  2021 N-CMAPSS_DS01-005.h5
## -rw-r--r-- 1 prasanna prasanna 2337M Jan 13  2021 N-CMAPSS_DS02-006.h5
## -rw-r--r-- 1 prasanna prasanna 3523M Jan 13  2021 N-CMAPSS_DS03-012.h5
## -rw-r--r-- 1 prasanna prasanna 3579M Jan 13  2021 N-CMAPSS_DS04.h5
## -rw-r--r-- 1 prasanna prasanna 2479M Jan 12  2021 N-CMAPSS_DS05.h5
## -rw-r--r-- 1 prasanna prasanna 2432M Jan 13  2021 N-CMAPSS_DS06.h5
## -rw-r--r-- 1 prasanna prasanna 2589M Jan 13  2021 N-CMAPSS_DS07.h5
## -rw-r--r-- 1 prasanna prasanna 3087M Jan 13  2021 N-CMAPSS_DS08a-009.h5
## -rw-r--r-- 1 prasanna prasanna 2302M Jan 13  2021 N-CMAPSS_DS08c-008.h5
## -rw-r--r-- 1 prasanna prasanna 2752M Jan 13  2021 N-CMAPSS_DS08d-010.h5
## -rw-r--r-- 1 prasanna prasanna    5M Aug  1 16:30 N-CMAPSS_Example_data_loading_and_exploration.ipynb
## -rw-r--r-- 1 prasanna prasanna    1M Jan 14  2021 Run_to_Failure_Simulation_Under_Real_Flight_Conditions_Dataset.pdf
```

We have about 25G of data over 10 files in h5 format. 
