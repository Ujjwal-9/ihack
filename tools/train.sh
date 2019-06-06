#!/bin/bash

source activate retina
retina-train csv "./data/annotations.csv" "./data/class.csv"
