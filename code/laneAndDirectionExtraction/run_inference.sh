#!/usr/bin/env bash
# Derek Yuen
# 2026-08

# previously known as infer_pipeline.py in the Unofficial version
# rewritten in bash

#set -e      # exit for any failure

export TF_CPP_MIN_LOG_LEVEL=3

echo ""
echo "*** run_inference.sh for laneAndDirectionExtraction ***"

for i in 0 5 6 11 12 17 18 22 25 28 31; do
    validation_data="validation_laneExtraction_run1_640_resnet34v3_500ep"
    inputfile="../../dataset/sat_${i}.jpg"
    outputfolder="./output/${i}"
    model="resnet34v3"

    if [ -f "$inputfile" ]; then
        echo '------'
        echo "${i}: ${inputfile}, saving to: ${outputfolder}, model: ${model}"

        # rm -f "${outputfolder}"/*
        mkdir -vp "${outputfolder}"

        # copy or replace the sdmap file into the expected filename format; overwrite if necessary
        cp -f "${validation_data}/sdmap${i}.jpg" "../../dataset/sdmap_${i}.jpg" ; echo ""

        echo "Run inference for the first stage and extract the graph"
        python3 infer.py "${inputfile}" "${outputfolder}" "${model}" || { echo "infer.py failed for ${i}"; exit 1; }


        echo "Turning segmentation to graph (${outputfolder}/seg.png)"
        python3 segtograph/setograph.py "${outputfolder}/seg.png" 64 "${outputfolder}/graph.p" || { echo "setograph.py failed for ${i}"; exit 1; }


        echo "Extract directions and create ways.json file (${outputfolder})"
        python3 infer_direction.py "${outputfolder}/direction.png" "${outputfolder}/graph.p" "${outputfolder}" || { echo "infer_direction.py failed for ${i}"; exit 1; }

        # cleanup
        rm -f "../../dataset/sdmap_${i}.jpg"

    else
        echo "" ; echo "ERROR: ${inputfile} missing"
    fi

done