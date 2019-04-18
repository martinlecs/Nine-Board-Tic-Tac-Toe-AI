#!/bin/sh
./src/servt -p 54321 &
./src/lookt.mac -p 54321 &
/Users/martinle/.local/share/virtualenvs/game-0pSFqvFS/bin/python3 player/agent.py -p 54321
