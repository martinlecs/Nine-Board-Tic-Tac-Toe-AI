#!/bin/sh
./src/servt -p 54321 &
./src/lookt -p 54321 &
python3 agent.py -p 54321
