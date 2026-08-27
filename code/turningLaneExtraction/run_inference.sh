#!/usr/bin/env bash
# Derek Yuen
# 2026-08

# previously known as run_test.bash / infer_pipeline.py in the Unofficial version
# https://github.com/TonyXuQAQ/UnofficialLaneExtraction/blob/master/code/turningLaneExtraction/run_test.bash
# https://github.com/TonyXuQAQ/UnofficialLaneExtraction/blob/master/code/turningLaneExtraction/infer_pipeline.py
# rewritten in bash


for i in 0 5 6 11 12 17 18 22 25 28 31; do

    inputfile="../../dataset/sat_${i}.jpg"
    outputfolder="./output/${i}"
    model="resnet34v3"

    echo "${inputfile} -> ${outputfolder}, model: ${model}"

    # rm -f "${outputfolder}"/*
    mkdir -vp "${outputfolder}"

    # copy or replace the sdmap file into the expected filename format; overwrite if necessary
    cp -fv "validation_laneExtraction_run1_640_resnet34v3_500ep/sdmap${i}.jpg" "../../dataset/sdmap_${i}.jpg"

    echo "run inference for the first stage and extract the graph"
    #python3 infer.py "${inputfile}" "${outputfolder}" "${model}" ; echo ""
    python infer_link_v4.py "$inputfile" "$outputfolder/direction.png" "$outputfolder/graph.p" "$outputfolder"

    echo "Turning segmentation to graph"
    python3 segtograph/setograph.py "${outputfolder}/seg.png" 64 "${outputfolder}/graph.p" ; echo ""


    echo "extract directions and create ways.json file"
    python3 infer_direction.py "${outputfolder}/direction.png" "${outputfolder}/graph.p" "${outputfolder}" ; echo ""

    rm -fv "../../dataset/sdmap_${i}.jpg" ; echo ""

done