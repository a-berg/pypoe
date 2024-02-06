# pypoe

Python library and cli (coming soon!) for all things Path of Exile.

Path of Exile has spawned over the years a number of peripheric tools to provide functionality the community felt as lacking
in the base game. Most of these tools have many years now.

In particular, this project is born due to my frustration with the venerable Vorici's Chromatic Calculator. The info is
good, but as soon as I wondered what the real cost of colouring my items would be (instead of just having an average and
standard deviation) I started to find having to deal with HTML format cumbersome. So I decided to use Python to create
a calculator that does  the same job as the Vorici calculator, except I have access to `scipy` to compute percentiles
exactly. So now you can know what the .66, .80, .90, .95 and .99 percentiles for your desired colors are (spoiler: it's
much more expensive than you think).

**Disclaimer:** This is an extremely alpha library with very little functionality, developed by 1 guy in his free time. 

## Features

* chromatic orb cost computation with percentiles.

That's it for now!
