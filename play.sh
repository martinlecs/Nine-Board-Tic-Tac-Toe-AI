#!/bin/sh
./src/servt -p 54321 -n 10 &
/Users/martinle/.local/share/virtualenvs/game-0pSFqvFS/bin/python3 agent.py -p 54321 &
./src/lookt.mac -p 54321 -d 16 &
