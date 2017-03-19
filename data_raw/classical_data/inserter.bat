rem for %%i in (*.mid) do c:/Programmi/ActivePerl/bin/perl.exe inserter.pl comment.txt %%i

for /F "tokens=*" %%i in ('dir /B /S *.mid') do c:/Programmi/ActivePerl/bin/perl.exe inserter.pl comment.txt %%i