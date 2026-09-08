## How the strongest agents build

Averaged over the 6,646 games the public ladder played in the last 10
days, across the 12 agents a Bradley-Terry fit over that stretch rates
highest. The window is not a sample: the field turns over completely inside a
fortnight, so a table averaged over the whole corpus describes a blend of
fields, most of which no longer plays. None of these agents publishes a kernel,
so their games are the only view of them there is, and none of them is in the
pool you are being scored against.

Read down a column and you have what the top of the ladder holds on that day.
Every row here is also a column of your own day tables below, so you can put
your number beside theirs directly.

This is not a target to hit and not a rule of the game. It is what the
strongest quarter of a 185-agent field happens to do, and any of it can be
beaten -- but where you differ from it sharply and lose, this is the first
place to look.

| day           | d0   | d1   | d2   | d3   | d4   | d5   | d6   | d8    | d10    | d12    | d14    | d17    | d20    | d23    | d25    | d27    | d29    |
| ------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ----- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| bank          | 39   | 206  | 296  | 221  | 419  | 402  | 538  | 1,004 | 11,665 | 14,925 | 20,915 | 36,884 | 54,738 | 68,516 | 75,726 | 83,937 | 97,383 |
| quadrants     | 1.0  | 1.0  | 1.0  | 1.2  | 1.2  | 1.3  | 2.0  | 2.3   | 2.5    | 3.0    | 3.0    | 3.0    | 3.0    | 3.0    | 3.0    | 3.0    | 3.0    |
| planted tiles | 18.0 | 18.4 | 18.7 | 19.1 | 16.9 | 19.5 | 31.6 | 39.4  | 42.1   | 58.3   | 57.6   | 57.5   | 57.5   | 56.0   | 55.4   | 51.1   | 3.9    |
| fertilised    | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0   | 0.1    | 1.7    | 7.4    | 11.3   | 18.8   | 10.7   | 15.5   | 9.1    | 0.9    |
| animals       | 4.5  | 4.5  | 5.1  | 5.2  | 5.9  | 6.4  | 8.1  | 11.5  | 13.6   | 15.0   | 15.3   | 15.1   | 14.6   | 14.0   | 13.7   | 13.4   | 11.6   |
| hands         | 5.3  | 3.6  | 4.6  | 5.2  | 4.6  | 5.2  | 7.9  | 9.4   | 11.0   | 10.4   | 10.9   | 11.4   | 11.6   | 11.6   | 11.4   | 11.3   | 10.1   |
| seed in store | 0.3  | 0.1  | 1.1  | 1.3  | 4.5  | 5.6  | 4.2  | 6.8   | 8.1    | 7.6    | 7.7    | 8.8    | 10.0   | 9.1    | 7.4    | 5.4    | 5.1    |
| shed          | 3.3  | 4.6  | 1.8  | 2.9  | 8.5  | 5.4  | 9.8  | 9.1   | 37.3   | 18.9   | 12.5   | 11.2   | 12.0   | 8.8    | 7.5    | 7.2    | 0.8    |
| weeds         | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.1  | 0.0  | 0.0   | 0.0    | 0.0    | 0.0    | 0.1    | 0.2    | 1.0    | 1.4    | 1.4    | 6.6    |

What the same games say when the two sides of each are compared directly --
every quantity crossed with every day, and these are the ones that separate
the stronger agent from the weaker most sharply:

- **land_orders**, day 3: the stronger side holds more, in 99% of 216 games where the two differed.
- **quadrants**, day 3: the stronger side holds more, in 99% of 202 games where the two differed.
- **land_orders**, day 8: the stronger side holds more, in 98% of 460 games where the two differed.
- **animal_units**, day 1: the stronger side holds more, in 97% of 600 games where the two differed.
- **land_orders**, day 4: the stronger side holds more, in 97% of 224 games where the two differed.
- **animal_units**, day 0: the stronger side holds more, in 97% of 601 games where the two differed.
- **pens**, day 1: the stronger side holds more, in 97% of 601 games where the two differed.
- **pens**, day 0: the stronger side holds more, in 96% of 547 games where the two differed.
