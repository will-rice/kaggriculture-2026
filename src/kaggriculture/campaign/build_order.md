## How the strongest agents build

Averaged over the 6,642 games the public ladder played in the last 10
days, across the 3 agents that share the strongest opening on it:
**second quadrant day 3, third day 8**, mean rating +2.19.

Those agents rather than the top of the table, because a mean across the top of
the table is a mean across agents doing different things. Measured 2026-09-09:
the top twelve hold 1.16 quadrants on day three, which is 84% of them holding
one and 16% holding two. No agent holds 1.16 quadrants. Averaged, the group
that takes land on day three -- the three best agents on the ladder, 0.9
log-odds clear of fourth -- is blended with the group that waits until day six,
and the number that separates first place from fourth is deleted.

The opening is a property of the agent and not of the game: within one agent the
day it takes its second quadrant varies by a tenth of a day, while between
agents it ranges from three to six. So this is a strategy, and it is followable
in a way the mean was not.

The window is not a sample either: the field turns over completely inside a
fortnight, so a table over the whole corpus describes a blend of fields, most of
which no longer plays. None of these agents publishes a kernel, so their games
are the only view of them there is, and none is in the pool you are scored
against.

Read down a column and you have what the top of the ladder holds on that day.
Every row here is also a column of your own day tables below, so you can put
your number beside theirs directly.

This is not a target to hit and not a rule of the game. It is what the
strongest quarter of a 185-agent field happens to do, and any of it can be
beaten -- but where you differ from it sharply and lose, this is the first
place to look.

| day           | d0   | d1   | d2   | d3   | d4  | d5   | d6   | d8    | d10   | d12   | d14    | d17    | d20    | d23    | d25    | d27    | d29     |
| ------------- | ---- | ---- | ---- | ---- | --- | ---- | ---- | ----- | ----- | ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------- |
| bank          | 40   | 626  | 996  | 317  | 757 | 572  | 778  | 1,502 | 3,721 | 9,296 | 15,519 | 35,002 | 56,226 | 71,990 | 80,730 | 90,031 | 106,246 |
| quadrants     | 1.0  | 1.0  | 1.0  | 2.0  | 2.0 | 2.0  | 2.0  | 2.9   | 3.0   | 3.0   | 3.0    | 3.0    | 3.0    | 3.0    | 3.0    | 3.0    | 3.0     |
| planted tiles | 19.0 | 19.0 | 19.0 | 20.1 | 7.6 | 17.1 | 33.5 | 43.5  | 58.7  | 59.4  | 57.6   | 57.6   | 56.9   | 55.6   | 53.3   | 53.7   | 4.2     |
| fertilised    | 0.0  | 0.0  | 0.0  | 0.0  | 0.0 | 0.0  | 0.0  | 0.0   | 0.0   | 0.3   | 5.6    | 11.3   | 20.6   | 17.8   | 15.2   | 15.4   | 1.5     |
| animals       | 6.0  | 6.0  | 6.0  | 6.0  | 6.1 | 8.7  | 9.1  | 10.8  | 14.1  | 14.5  | 15.3   | 15.9   | 16.3   | 16.4   | 16.4   | 16.3   | 13.4    |
| hands         | 7.0  | 4.3  | 7.0  | 7.0  | 7.0 | 7.0  | 7.3  | 9.2   | 12.0  | 12.0  | 12.0   | 12.0   | 12.0   | 12.0   | 12.0   | 12.0   | 12.0    |
| seed in store | 0.0  | 0.0  | 0.0  | 0.6  | 0.8 | 0.4  | 0.8  | 8.5   | 0.7   | 0.5   | 0.5    | 0.4    | 0.6    | 0.8    | 1.1    | 0.8    | 0.8     |
| shed          | 6.0  | 0.0  | 2.2  | 0.3  | 0.3 | 2.4  | 1.4  | 3.2   | 11.1  | 14.2  | 19.3   | 9.2    | 5.4    | 6.5    | 2.9    | 3.6    | 0.2     |
| weeds         | 0.0  | 0.0  | 0.0  | 0.0  | 0.1 | 0.3  | 0.1  | 0.0   | 0.0   | 0.0   | 0.0    | 0.0    | 0.0    | 0.2    | 0.3    | 0.3    | 1.0     |

What the same games say when the two sides of each are compared directly --
every quantity crossed with every day, and these are the ones that separate
the stronger agent from the weaker most sharply:

- **fertilised**, day 6: the stronger side holds more, in 93% of 268 games where the two differed.
- **fertilised**, day 7: the stronger side holds more, in 93% of 275 games where the two differed.
- **fertilised**, day 8: the stronger side holds more, in 91% of 235 games where the two differed.
- **land_orders**, day 8: the stronger side holds more, in 89% of 620 games where the two differed.
- **hungry_worst**, day 2: the stronger side holds more, in 88% of 275 games where the two differed.
- **quadrants**, day 3: the stronger side holds more, in 88% of 292 games where the two differed.
- **quadrants**, day 8: the stronger side holds more, in 88% of 577 games where the two differed.
- **land_orders**, day 5: the stronger side holds more, in 87% of 639 games where the two differed.
