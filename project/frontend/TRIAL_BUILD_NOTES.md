# Trial Build Notes

## What you can use right now

Double-click:

`StartIdleBrainTrial.bat`

This starts the desktop launcher from the local Python environment and opens:

`http://127.0.0.1:8787`

## Why there is no EXE yet

The PyInstaller build reached the final EXE step, but Windows Defender blocked it with:

`WinError 225`

That means the current blocker is OS security policy, not an IdleBrain import/runtime error.

## Status of the attempted build

- Dependency install: passed
- PyInstaller analysis: passed
- Resource collection: passed
- EXE finalization: blocked by Defender

## If you want a real EXE later

The next options are:

1. Add a Defender exclusion for the build folder and rebuild.
2. Rework the packaging strategy to reduce antivirus false positives.
3. Build on another machine or clean CI runner.
