## How the strongest agents build

Averaged over the 6,642 games the public ladder played in the last 10
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

| day           | d0   | d1   | d2   | d3   | d4   | d5   | d6   | d8   | d10    | d12    | d14    | d17    | d20    | d23    | d25    | d27    | d29    |
| ------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| bank          | 46   | 202  | 298  | 220  | 418  | 391  | 570  | 977  | 11,195 | 14,930 | 20,743 | 36,927 | 54,866 | 69,036 | 76,394 | 84,745 | 98,359 |
| quadrants     | 1.0  | 1.0  | 1.0  | 1.2  | 1.2  | 1.3  | 2.0  | 2.3  | 2.5    | 3.0    | 3.0    | 3.0    | 3.0    | 3.0    | 3.0    | 3.0    | 3.0    |
| planted tiles | 17.8 | 18.4 | 18.6 | 18.8 | 16.6 | 19.1 | 31.4 | 39.7 | 42.7   | 58.2   | 57.2   | 57.3   | 56.7   | 55.2   | 54.5   | 50.3   | 4.1    |
| fertilised    | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.1  | 0.1    | 2.3    | 8.4    | 12.0   | 19.1   | 11.8   | 16.6   | 10.4   | 1.0    |
| animals       | 4.5  | 4.5  | 5.0  | 5.2  | 5.8  | 6.4  | 8.0  | 11.5 | 13.8   | 15.3   | 15.5   | 15.5   | 15.1   | 14.5   | 14.1   | 13.7   | 11.6   |
| hands         | 5.3  | 3.7  | 4.7  | 5.2  | 4.8  | 5.3  | 7.8  | 9.4  | 11.0   | 10.1   | 10.7   | 11.3   | 11.6   | 11.6   | 11.3   | 11.1   | 10.0   |
| seed in store | 0.5  | 0.2  | 1.2  | 1.4  | 4.5  | 5.6  | 4.3  | 6.6  | 7.8    | 6.7    | 6.3    | 6.6    | 7.3    | 6.7    | 5.7    | 4.5    | 4.2    |
| shed          | 3.5  | 4.4  | 2.0  | 2.9  | 8.4  | 5.6  | 8.4  | 8.0  | 34.3   | 14.8   | 10.2   | 8.7    | 10.1   | 7.4    | 6.5    | 6.5    | 1.1    |
| weeds         | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.1  | 0.0  | 0.0  | 0.0    | 0.0    | 0.0    | 0.1    | 0.2    | 1.4    | 1.9    | 2.0    | 7.7    |

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
