#!/usr/bin/perl
#
# $Id: inserter,v 0.1 2005/03/16 22:33:44 nicb Exp $
#
# Usage: ./comment <text file> <file MIDI 1> <...>\n"
#
use strict;

use IO::File;
use MIDI::Simple;
use MIDI::Opus;

die("Usage: $0 <text file> <file MIDI 1> <...>\n") unless ($ARGV[0]);

my $textfh = IO::File::new();
my $text = "";
my $line = "";

if ($textfh->open($ARGV[0], 'r'))
{
	while($line = <$textfh>)
	{
		$text = sprintf("%s%s", $text, $line);
	}
}

chop($text);

for (my $i = 1; $i < scalar(@ARGV); ++$i)
{
	my $file = $ARGV[$i];
	my $output = $file;
	my $midifile = MIDI::Opus->new({ "from_file" => $file });
	my $tracks_hr = $midifile->tracks_r();
	my $new_track = MIDI::Track->new();
	my $events_hr = $new_track->events_r();

	my $copyright = $text;
	my $copyrighter = substr($file,rindex($file,"(")); #cut before )
	$copyrighter = substr($copyrighter,0,rindex($copyrighter,".")); #cut after .

#print "$copyrighter";
	$copyright =~ s/CNTRB/$copyrighter/g;
#print "$copyright";
	push(@{$events_hr}, ['copyright_text_event', 1, "$copyright"]);

	$new_track->events_r($events_hr);

	#if a comment is already present... otherwise comment the following line!
	pop(@{$tracks_hr});

	push(@{$tracks_hr}, $new_track);

	$midifile->write_to_file($output);

	print "$output\n";
}

exit(0);
