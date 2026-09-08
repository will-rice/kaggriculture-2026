## How the strongest agents build

Averaged over the 16,292 recorded games of the public ladder, across the 25
agents a Bradley-Terry fit over the whole field rates highest. None of them
publishes a kernel, so their games are the only view of them there is, and none
of them is in the pool you are being scored against.

Read down a column and you have what the top of the ladder holds on that day.
Every row here is also a column of your own day tables below, so you can put
your number beside theirs directly.

This is not a target to hit and not a rule of the game. It is what the
strongest quarter of a 185-agent field happens to do, and any of it can be
beaten -- but where you differ from it sharply and lose, this is the first
place to look.

| day           | d0   | d1   | d2   | d3   | d4   | d5   | d6   | d8   | d10    | d12    | d14    | d17    | d20    | d23    | d25    | d27    | d29    |
| ------------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| bank          | 51   | 119  | 169  | 242  | 285  | 306  | 447  | 769  | 11,295 | 14,279 | 19,875 | 36,184 | 53,030 | 66,333 | 73,559 | 81,633 | 95,647 |
| quadrants     | 1.0  | 1.0  | 1.0  | 1.1  | 1.1  | 1.4  | 2.0  | 2.4  | 2.5    | 3.0    | 3.0    | 3.0    | 3.0    | 3.0    | 3.0    | 3.0    | 3.0    |
| planted tiles | 16.3 | 17.7 | 19.2 | 19.3 | 18.4 | 20.5 | 31.0 | 40.1 | 43.5   | 58.9   | 57.9   | 57.4   | 57.5   | 56.6   | 55.6   | 48.1   | 4.2    |
| fertilised    | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.2  | 1.1  | 0.5  | 0.2    | 3.4    | 8.1    | 11.9   | 18.0   | 10.4   | 13.4   | 8.6    | 1.0    |
| animals       | 4.4  | 4.5  | 5.0  | 5.0  | 5.6  | 5.9  | 7.9  | 11.3 | 13.1   | 14.3   | 14.9   | 15.1   | 15.0   | 14.9   | 14.7   | 14.6   | 12.1   |
| hands         | 4.7  | 3.7  | 4.2  | 4.7  | 4.2  | 5.3  | 7.8  | 9.0  | 11.0   | 10.5   | 11.0   | 11.6   | 11.7   | 11.7   | 11.5   | 11.4   | 10.9   |
| seed in store | 0.3  | 0.2  | 0.7  | 0.9  | 3.7  | 5.2  | 5.9  | 7.8  | 8.8    | 7.3    | 8.1    | 9.7    | 10.9   | 9.6    | 8.5    | 6.0    | 5.6    |
| shed          | 3.6  | 3.6  | 2.6  | 4.0  | 8.4  | 5.3  | 10.1 | 9.9  | 35.4   | 23.2   | 17.5   | 18.5   | 18.4   | 13.0   | 13.9   | 15.7   | 0.8    |
| weeds         | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.0  | 0.1    | 0.1    | 0.4    | 0.4    | 0.5    | 1.2    | 1.6    | 1.7    | 3.6    |

The shape of it, which the numbers above make hard to see all at once:

- **Capacity first, and early.** Land and quadrants by day five, animals from
  day zero, hands hired ahead of need. The strongest agents open their second
  and third quadrant two to three days before the rest of the field.
- **Poor on purpose until day twelve.** Their bank trails the field through the
  first ten days -- 447 against 1,380 on day six -- because it is in the
  ground. It crosses over around day twelve and finishes ahead, 95,647 against
  86,627.
- **Fertilizer is applied, not sold.** The weaker half of the field sells 1,787
  units of it a game; the strong sell 258 and buy more on top. On day five they
  hold fertilised tiles where the rest hold none, which is the single sharpest
  separation in the corpus: 100% of 313 paired games.
- **Seed goes in the ground.** From day six the strong carry about half the
  unplanted seed the rest do. They buy it and plant it rather than stockpiling.
- **They are net buyers.** After hiring, their busiest market activity is
  _buying_ wheat. Over a season they move 1,965 units against the field's
  4,998, and finish richer -- the weak field sells 2,859 units in the last six
  days alone.
